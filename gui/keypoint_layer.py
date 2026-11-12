"""Canvas layer for optional per-instance keypoints: drawing, placing, moving.

Inactive (draws nothing, consumes no events) until a project keypoint configuration
is set with ``set_config``. Keypoints live in the canvas's per-instance metadata under
``"keypoints"`` (see core/keypoints.py), so they save, load and sync with the rest
of the annotation without changes to the file format.

Mouse, with the canvas tool set to ``'keypoint'``:
  * left-click            place the next missing keypoint of the current bee; when it
                          is complete (or none is chosen), a click on a bee starts it
  * shift+click           place the next keypoint as hidden (occluded); on an existing
                          point, switch it between hidden and visible
  * drag a keypoint       move it (any visible bee)
  * right-click           menu: hidden/visible, delete point, clear bee, finish bee
  * double-click          same as a click (quick successive placements)
Keys: Esc finishes the bee, Backspace/Delete removes the hovered or last-placed point,
N moves to the next keypoint name, H toggles "place as hidden".
Undo/redo (``undo()``/``redo()``) cover keypoint edits on the current frame.
"""

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QMenu

from core import keypoints as kp

_BASE_COLORS = {
    'head': (255, 70, 70),
    'thorax': (255, 215, 0),
    'abdomen_tip': (60, 150, 255),
    'left_antenna_tip': (0, 220, 120),     # distinct left/right colours help keep sides consistent
    'right_antenna_tip': (255, 120, 220),
}
_PALETTE = [(0, 220, 120), (255, 120, 220), (255, 150, 40), (150, 110, 255), (0, 230, 230),
            (200, 255, 80), (255, 255, 255)]
Z_EDGE = 125
Z_POINT = 126


def keypoint_color(name, index):
    return _BASE_COLORS.get(name) or _PALETTE[index % len(_PALETTE)]


class KeypointLayer(QObject):
    """Keypoint display and editing for an ``ImageCanvas``."""

    state_changed = pyqtSignal()  # target bee, next keypoint, or its points changed

    RADIUS_PX = 5.5      # drawn radius in screen pixels (constant under zoom)
    HIT_PX = 11.0        # pick radius in screen pixels
    HISTORY_LIMIT = 200

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.config = None
        self.visible = True
        self.place_occluded = False
        self.target_id = 0
        self.active_name = None
        self.last_placed = None
        self.items = []
        self.undo_stack = []
        self.redo_stack = []
        self._drag = None
        self._hover = None
        canvas.instance_selected.connect(self._on_instance_selected)

    # ----- configuration and state -------------------------------------------------
    @property
    def enabled(self):
        return self.config is not None

    @property
    def names(self):
        return self.config.names if self.config else ()

    def set_config(self, config):
        """Turn the layer on with a ``KeypointConfig``, or off with None."""
        self.config = config
        self._drag = None
        self._hover = None
        if config is None:
            self.target_id = 0
            self.active_name = None
        elif self.active_name not in config.names:
            self.active_name = self._next_name(self.target_id)
        self.redraw()
        self.state_changed.emit()

    def set_visible(self, visible):
        self.visible = bool(visible)
        self.redraw()

    def set_place_occluded(self, occluded):
        self.place_occluded = bool(occluded)
        self.state_changed.emit()

    def reset_for_frame(self):
        """New frame/annotations: forget target, drag and the per-frame history."""
        self.clear_items()  # set_annotations([]) returns before any redraw
        self.target_id = 0
        self.active_name = self.names[0] if self.names else None
        self.last_placed = None
        self._drag = None
        self._hover = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.state_changed.emit()

    def on_tool_left(self):
        if self._drag is not None:
            self._finish_drag()
        self._hover = None

    def on_instance_renamed(self, old_id, new_id):
        if self.target_id == old_id:
            self.target_id = new_id
        for stack in (self.undo_stack, self.redo_stack):
            for step in stack:
                for i, (iid, before, after) in enumerate(step):
                    if iid == old_id:
                        step[i] = (new_id, before, after)

    # ----- data access -------------------------------------------------------------
    def _metadata(self, instance_id):
        return self.canvas.annotation_metadata.get(instance_id)

    def keypoints_for(self, instance_id):
        return kp.get_keypoints(self._metadata(instance_id))

    def _is_target_category(self, metadata):
        return bool(metadata) and metadata.get('category', 'bee') == self.config.category

    def _visible_instances(self):
        """{instance id: keypoints} for displayed instances of the configured category."""
        out = {}
        if not self.enabled:
            return out
        for instance_id, metadata in self.canvas.annotation_metadata.items():
            if not self._is_target_category(metadata):
                continue
            if not self.canvas.is_annotation_instance_visible(instance_id, self.config.category):
                continue
            points = kp.get_keypoints(metadata)
            if points:
                out[instance_id] = points
        return out

    def _overlays_shown(self):
        return self.visible and (self.canvas.show_segmentations or self.canvas.show_bboxes)

    def _store(self, changes, record=True):
        """Apply ``{instance_id: new keypoints}``; one undo step for the whole change."""
        step = []
        for instance_id, new_points in changes.items():
            metadata = self._metadata(instance_id)
            if metadata is None:
                continue
            before = kp.get_keypoints(metadata)
            after = kp.set_keypoints(metadata, new_points)
            if before != after:
                step.append((instance_id, before, after))
        if not step:
            return False
        if record:
            self.undo_stack.append(step)
            del self.undo_stack[:-self.HISTORY_LIMIT]
            self.redo_stack.clear()
        self.redraw()
        self.canvas.annotation_changed.emit()
        self.state_changed.emit()
        return True

    def _apply_step(self, step, use_after):
        for instance_id, before, after in step:
            metadata = self._metadata(instance_id)
            if metadata is not None:
                kp.set_keypoints(metadata, after if use_after else before)
        self.redraw()
        self.canvas.annotation_changed.emit()
        self.state_changed.emit()

    def undo(self):
        if not self.undo_stack:
            return False
        step = self.undo_stack.pop()
        self.redo_stack.append(step)
        self._apply_step(step, use_after=False)
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        step = self.redo_stack.pop()
        self.undo_stack.append(step)
        self._apply_step(step, use_after=True)
        return True

    # ----- editing operations (also used by the toolbar) ---------------------------
    def _next_name(self, instance_id, after=None):
        if not self.enabled:
            return None
        return kp.next_unlabelled(self.keypoints_for(instance_id) if instance_id else {},
                                  self.names, after=after)

    def set_target(self, instance_id, select=True):
        """Make ``instance_id`` the bee being labelled (0 to finish/clear)."""
        metadata = self._metadata(instance_id) if instance_id else None
        if instance_id and not (self.enabled and self._is_target_category(metadata)):
            return False
        self.target_id = int(instance_id or 0)
        self.last_placed = None
        self.active_name = self._next_name(self.target_id) if self.target_id else (
            self.names[0] if self.names else None)
        if select and self.target_id:
            self.canvas.highlight_instance(self.target_id, self.config.category)
        self.redraw()
        self.state_changed.emit()
        return True

    def set_active_name(self, name):
        if name in self.names:
            self.active_name = name
            self.state_changed.emit()

    def advance_active_name(self):
        if not self.names:
            return
        if self.active_name in self.names:
            i = self.names.index(self.active_name)
            self.active_name = self.names[(i + 1) % len(self.names)]
        else:
            self.active_name = self.names[0]
        self.state_changed.emit()

    def place(self, instance_id, name, x, y, visibility=None):
        if visibility is None:
            visibility = kp.OCCLUDED if self.place_occluded else kp.VISIBLE
        points = kp.with_point(self.keypoints_for(instance_id), name, x, y, visibility)
        if self._store({instance_id: points}):
            self.last_placed = name
            if instance_id == self.target_id:
                self.active_name = self._next_name(instance_id, after=name)
            self.state_changed.emit()
            return True
        return False

    def delete_point(self, instance_id, name):
        changed = self._store({instance_id: kp.without_point(self.keypoints_for(instance_id), name)})
        if changed and instance_id == self.target_id:
            self.active_name = name
            if self.last_placed == name:
                self.last_placed = None
            self.state_changed.emit()
        return changed

    def toggle_visibility(self, instance_id, name):
        return self._store({instance_id: kp.with_toggled_visibility(self.keypoints_for(instance_id), name)})

    def clear_instance(self, instance_id):
        changed = self._store({instance_id: {}})
        if changed and instance_id == self.target_id:
            self.active_name = self.names[0] if self.names else None
            self.last_placed = None
            self.state_changed.emit()
        return changed

    def fill_from_markers(self, only_missing=True):
        """Set the configured marker keypoint (thorax) from each instance's ArUco centre."""
        if not self.enabled or not self.config.marker_keypoint:
            return 0
        name = self.config.marker_keypoint
        changes = {}
        for instance_id, metadata in self.canvas.annotation_metadata.items():
            if not self._is_target_category(metadata):
                continue
            center = kp.marker_center(metadata)
            points = kp.get_keypoints(metadata)
            if center is None or (only_missing and name in points):
                continue
            changes[instance_id] = kp.with_point(points, name, center[0], center[1], kp.VISIBLE)
        if changes and self._store(changes):
            if self.target_id in changes:
                self.active_name = self._next_name(self.target_id)
                self.state_changed.emit()
            return len(changes)
        return 0

    def count_marker_candidates(self):
        if not self.enabled or not self.config.marker_keypoint:
            return 0
        name = self.config.marker_keypoint
        return sum(
            1 for m in self.canvas.annotation_metadata.values()
            if self._is_target_category(m) and kp.marker_center(m) is not None
            and name not in kp.get_keypoints(m)
        )

    def labelled_instance_count(self):
        if not self.enabled:
            return 0
        return sum(1 for m in self.canvas.annotation_metadata.values()
                   if self._is_target_category(m) and kp.get_keypoints(m))

    # ----- hit testing -------------------------------------------------------------
    def _view_scale(self):
        scale = abs(self.canvas.transform().m11())
        return scale if scale > 1e-6 else 1.0

    def hit(self, scene_pos):
        """(instance_id, name) of the displayed keypoint under the cursor, or None."""
        if not (self.enabled and self._overlays_shown()):
            return None
        found = kp.nearest_keypoint(self._visible_instances(), scene_pos.x(), scene_pos.y(),
                                    self.HIT_PX / self._view_scale())
        return (found[0], found[1]) if found else None

    def _instance_at(self, x, y):
        category = self.config.category
        mask_array = self.canvas._get_mask_array_by_category(category)
        xi, yi = int(x), int(y)
        if mask_array is None or not (0 <= xi < mask_array.shape[1] and 0 <= yi < mask_array.shape[0]):
            return 0
        instance_id = int(mask_array[yi, xi])
        if instance_id <= 0 and self.canvas.editing_instance_id > 0 and self.canvas.editing_mask is not None:
            if (self.canvas.editing_instance_category == category and
                    self.canvas.editing_mask[yi, xi] > 0):
                instance_id = self.canvas.editing_instance_id
        if instance_id > 0 and self.canvas.is_annotation_instance_visible(instance_id, category):
            return instance_id
        return 0

    # ----- mouse and keys (called by the canvas while the tool is 'keypoint') -------
    def handle_mouse_press(self, event, scene_pos):
        if not self.enabled:
            return False
        button = event.button()
        if button == Qt.MouseButton.RightButton:
            self._show_context_menu(event, scene_pos)
            return True
        if button != Qt.MouseButton.LeftButton:
            return False
        image = self.canvas.current_image
        if image is None:
            return True
        h, w = image.shape[:2]
        x = min(max(scene_pos.x(), 0.0), w - 1.0)
        y = min(max(scene_pos.y(), 0.0), h - 1.0)

        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        picked = self.hit(scene_pos)
        if picked is not None:
            instance_id, name = picked
            if instance_id != self.target_id:
                self.set_target(instance_id)
            if shift:  # one point hidden/visible without a menu or a mode
                self.toggle_visibility(instance_id, name)
                return True
            self._drag = {'id': instance_id, 'name': name,
                          'before': self.keypoints_for(instance_id), 'moved': False}
            return True

        target_ok = bool(self.target_id) and self._is_target_category(self._metadata(self.target_id))
        if not target_ok or self.active_name is None:
            candidate = self._instance_at(x, y)
            if not candidate and not target_ok:
                selected = self.canvas.selected_mask_idx
                if selected > 0 and self._is_target_category(self._metadata(selected)):
                    candidate = selected
            if not candidate:
                return True  # clicked off any bee with nothing to place
            if candidate != self.target_id:
                self.set_target(candidate)
            if self.active_name is None:
                return True  # bee already complete: just selected it
        hidden = self.place_occluded or shift
        self.place(self.target_id, self.active_name, x, y, kp.OCCLUDED if hidden else kp.VISIBLE)
        return True

    def handle_mouse_move(self, event, scene_pos):
        if not self.enabled:
            return False
        if self._drag is not None:
            image = self.canvas.current_image
            h, w = image.shape[:2] if image is not None else (1, 1)
            x = min(max(scene_pos.x(), 0.0), w - 1.0)
            y = min(max(scene_pos.y(), 0.0), h - 1.0)
            metadata = self._metadata(self._drag['id'])
            if metadata is not None:
                points = self.keypoints_for(self._drag['id'])
                old = points.get(self._drag['name'], [x, y, kp.VISIBLE])
                kp.set_keypoints(metadata, kp.with_point(points, self._drag['name'], x, y, old[2]))
                self._drag['moved'] = True
                self.redraw()
            return True
        hover = self.hit(scene_pos)
        if hover != self._hover:
            self._hover = hover
            self.canvas.viewport().setCursor(
                Qt.CursorShape.SizeAllCursor if hover else Qt.CursorShape.CrossCursor)
        return False

    def handle_mouse_release(self, event, scene_pos):
        if self._drag is None:
            return False
        self._finish_drag()
        return True

    def _finish_drag(self):
        drag, self._drag = self._drag, None
        if not drag or not drag['moved']:
            return
        metadata = self._metadata(drag['id'])
        if metadata is None:
            return
        after = kp.get_keypoints(metadata)
        if after != drag['before']:
            self.undo_stack.append([(drag['id'], drag['before'], after)])
            del self.undo_stack[:-self.HISTORY_LIMIT]
            self.redo_stack.clear()
            self.canvas.annotation_changed.emit()
            self.state_changed.emit()

    def handle_key(self, event):
        if not self.enabled:
            return False
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.set_target(0, select=False)
            return True
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            if self._hover is not None:
                self.delete_point(*self._hover)
                self._hover = None
            elif self.target_id and self.last_placed:
                self.delete_point(self.target_id, self.last_placed)
            return True
        if key == Qt.Key.Key_N:
            self.advance_active_name()
            return True
        if key == Qt.Key.Key_H:
            self.set_place_occluded(not self.place_occluded)
            return True
        return False

    def _show_context_menu(self, event, scene_pos):
        menu = QMenu(self.canvas)
        picked = self.hit(scene_pos)
        if picked is not None:
            instance_id, name = picked
            point = self.keypoints_for(instance_id).get(name)
            label = 'Mark Visible' if point and point[2] == kp.OCCLUDED else 'Mark Hidden (occluded)'
            a = QAction(f'{label}: {name} (bee {instance_id})', menu)
            a.triggered.connect(lambda _=False, i=instance_id, n=name: self.toggle_visibility(i, n))
            menu.addAction(a)
            a = QAction(f'Delete {name} (bee {instance_id})', menu)
            a.triggered.connect(lambda _=False, i=instance_id, n=name: self.delete_point(i, n))
            menu.addAction(a)
            menu.addSeparator()
            context_id = instance_id
        else:
            context_id = self.target_id or self._instance_at(scene_pos.x(), scene_pos.y())
        if context_id:
            if context_id == self.target_id:
                a = QAction(f'Finish Bee {context_id}', menu)
                a.triggered.connect(lambda _=False: self.set_target(0, select=False))
            else:
                a = QAction(f'Label Bee {context_id}', menu)
                a.triggered.connect(lambda _=False, i=context_id: self.set_target(i))
            menu.addAction(a)
            a = QAction(f'Clear All Keypoints of Bee {context_id}', menu)
            a.setEnabled(bool(self.keypoints_for(context_id)))
            a.triggered.connect(lambda _=False, i=context_id: self.clear_instance(i))
            menu.addAction(a)
        if menu.isEmpty():
            a = QAction('Click a bee to start its keypoints', menu)
            a.setEnabled(False)
            menu.addAction(a)
        pos = event.globalPosition().toPoint() if hasattr(event, 'globalPosition') else event.globalPos()
        menu.exec(pos)

    def _on_instance_selected(self, instance_id, category):
        if (self.enabled and self.canvas.current_tool == 'keypoint' and
                category == self.config.category and instance_id != self.target_id):
            self.set_target(instance_id, select=False)

    # ----- drawing -----------------------------------------------------------------
    def clear_items(self):
        scene = self.canvas.scene
        for item in self.items:
            try:
                if item.scene() is scene:
                    scene.removeItem(item)
            except RuntimeError:
                pass  # already deleted with the scene
        self.items = []

    def forget_items(self):
        """Drop references after the scene was cleared elsewhere."""
        self.items = []

    def redraw(self):
        self._draw_items()
        if self._drag is None:
            self.state_changed.emit()  # e.g. counts after a frame load; not while dragging

    def _draw_items(self):
        self.clear_items()
        if not (self.enabled and self._overlays_shown()):
            return
        instances = self._visible_instances()
        if not instances:
            return
        scene = self.canvas.scene
        r = self.RADIUS_PX
        index = {name: i for i, name in enumerate(self.names)}
        for instance_id, points in instances.items():
            is_target = instance_id == self.target_id
            rgb = self.canvas.display_color_for_instance(instance_id, self.config.category)
            edge_pen = QPen(QColor(*rgb[:3]) if rgb else QColor(255, 255, 255))
            edge_pen.setCosmetic(True)
            edge_pen.setWidthF(2.5 if is_target else 1.6)
            for a, b in self.config.edges:
                if a in points and b in points:
                    line = QGraphicsLineItem(points[a][0], points[a][1], points[b][0], points[b][1])
                    line.setPen(edge_pen)
                    line.setZValue(Z_EDGE)
                    scene.addItem(line)
                    self.items.append(line)
            for name, (x, y, v) in points.items():
                color = QColor(*keypoint_color(name, index.get(name, len(index))))
                item = QGraphicsEllipseItem(QRectF(-r, -r, 2 * r, 2 * r))
                item.setPos(QPointF(x, y))
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                outline = QPen(QColor(255, 255, 255) if is_target else QColor(0, 0, 0))
                outline.setWidthF(2.0 if is_target else 1.2)
                if v == kp.OCCLUDED:
                    outline = QPen(color)
                    outline.setWidthF(2.0)
                    outline.setStyle(Qt.PenStyle.DashLine)
                    item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                else:
                    item.setBrush(QBrush(color))
                item.setPen(outline)
                item.setZValue(Z_POINT)
                item.setToolTip(f"bee {instance_id}: {name}{' (hidden)' if v == kp.OCCLUDED else ''}")
                scene.addItem(item)
                self.items.append(item)
