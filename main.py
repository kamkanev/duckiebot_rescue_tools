#!/usr/bin/env python3
"""Simple GUI app: left sidebar to pick saved maps, center canvas shows image and graph.
Click two spots to choose start/end and press SPACE to run/step A*; R resets.
"""
import os
import sys
import json
import math
import pygame
import numpy as np

# Ensure repo root on path so utils.graph imports work
repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ''))
if repo_root not in sys.path:
	sys.path.insert(0, repo_root)

try:
	from utils.graph.AStar import Spot, AStarGraph, AStar
except Exception as e:
	print('Failed importing A* classes:', e)
	Spot = None
	AStarGraph = None
	AStar = None

from llm.route_instructions import parse_text_to_steps, write_turns_bin

pygame.init()

# Window layout
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SIDEBAR_W = 300
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (100, 100, 100)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 120, 255)
YELLOW = (255, 200, 0)
PURPLE = (138, 43, 226)
ORANGE = (255, 165, 0)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Map + Graph Viewer')
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 16)
title_font = pygame.font.SysFont('Arial', 20, bold=True)


class App:
	def __init__(self):
		self.saves_dir = os.path.join(repo_root, 'mapeditor', 'saves')
		self.available = self._get_available_graphs()
		self.selected_idx = 0

		self.graph = None
		self.astar = None
		self.start_spot = None
		self.end_spot = None
		self.algorithm_running = False
		self.algorithm_done = False

		self.image = None
		self.image_rect = None
		self.graph_offset_x = 0
		self.graph_offset_y = 0
		self.graph_scale = 1.0
		# cached graph bbox used for scaling to image
		self.graph_min_x = 0
		self.graph_min_y = 0
		# per-spot visual snap offsets (screen-space) to align with image roads
		self.snap_offsets = {}
		self.hover_spot = None

		# whether to render full graph (we'll hide by default and show only path/start/end)
		self.show_graph = False

		# runtime nudges for quick alignment checks
		self.image_shift_x = 0
		self.image_shift_y = 0
		self.graph_extra_x = 0
		self.graph_extra_y = 0

		# turn list scroll offset for sidebar
		self.turns_scroll = 0

		# Chat/LLM route input state
		self.text_path = []
		self.chat_active = False
		self.chat_text = ""
		self.chat_caret = 0
		self.chat_view_start = 0
		self.chat_message = ""
		self.chat_message_timer = 0
		self.caret_blink_ms = 500
		self.caret_visible = True
		self.last_caret_toggle = pygame.time.get_ticks()

	def _get_available_graphs(self):
		graphs = []
		if os.path.isdir(self.saves_dir):
			for name in os.listdir(self.saves_dir):
				full = os.path.join(self.saves_dir, name)
				if os.path.isdir(full):
					# accept if graph.json or <name>.json exists
					p1 = os.path.join(full, 'graph.json')
					p2 = os.path.join(full, f'{name}.json')
					if os.path.isfile(p1) or os.path.isfile(p2):
						graphs.append(name)
		return sorted(graphs)

	def load_graph(self, name):
		save_root = os.path.join(self.saves_dir, name)
		graph_path = os.path.join(save_root, 'graph.json')
		legacy_path = os.path.join(save_root, f'{name}.json')
		path = graph_path if os.path.isfile(graph_path) else legacy_path
		if not os.path.isfile(path):
			print('No graph file for', name)
			return False
		try:
			with open(path, 'r') as f:
				data = json.load(f)
			spots_data = data.get('spots', [])
			neigh_data = data.get('neighbors', [])
			costs_data = data.get('weights', [])

			new_spots = []
			for sd in spots_data:
				x = sd.get('x', 0)
				y = sd.get('y', 0)
				size = sd.get('size', 5)
				is_wall = sd.get('isWall', False)
				new_spots.append(Spot(x, y, size, is_wall))

			self.graph = AStarGraph(new_spots)
			# restore neighbors
			for i, neigh_list in enumerate(neigh_data):
				if i >= len(self.graph.spots):
					break
				# clear existing neighbors and costs
				self.graph.spots[i].neighbors = []
				self.graph.spots[i].costs = []

				# corresponding weights for this spot (if present)
				weights_for_spot = []
				if isinstance(costs_data, list) and i < len(costs_data):
					weights_for_spot = costs_data[i] if isinstance(costs_data[i], list) else []

				for k, j in enumerate(neigh_list):
					if 0 <= j < len(self.graph.spots):
						w = 1
						if k < len(weights_for_spot):
							try:
								w = float(weights_for_spot[k])
							except Exception:
								w = 1
						# use addNeighborWithCost to keep neighbors and costs in sync
						self.graph.spots[i].addNeighborWithCost(self.graph.spots[j], w)

			# load image if exists in graph_drawer/assets
			img_candidates = [
				os.path.join(repo_root, 'graph_drawer', 'assets', f'{name}.jpg'),
				os.path.join(save_root, f'{name}.jpg'),
			]
			img_path = None
			for c in img_candidates:
				if os.path.isfile(c):
					img_path = c
					break
			if img_path:
				try:
					# load image at its original size (no scaling)
					img = pygame.image.load(img_path).convert()
					iw, ih = img.get_size()
					self.image = img
					self.image_rect = self.image.get_rect()
					# center image in canvas area (to the right of sidebar)
					canvas_w = SCREEN_WIDTH - SIDEBAR_W
					canvas_h = SCREEN_HEIGHT
					self.image_rect.topleft = (SIDEBAR_W + (canvas_w - iw) // 2, (canvas_h - ih) // 2)
					# default small nudge: move image slightly for alignment
					self.image_shift_x = -5
					self.image_shift_y = -2
				except Exception as e:
					print('Failed to load image', e)
					self.image = None
					self.image_rect = None
			else:
				self.image = None
				self.image_rect = None

			# compute graph offset and scale to align graph to image (if image present)
			# compute graph bbox
			if self.graph and len(self.graph.spots) > 0:
				min_x = min(s.position.x for s in self.graph.spots)
				max_x = max(s.position.x for s in self.graph.spots)
				min_y = min(s.position.y for s in self.graph.spots)
				max_y = max(s.position.y for s in self.graph.spots)
				self.graph_min_x = min_x
				self.graph_min_y = min_y
				graph_w = max(1.0, max_x - min_x)
				graph_h = max(1.0, max_y - min_y)

				canvas_w = SCREEN_WIDTH - SIDEBAR_W
				canvas_h = SCREEN_HEIGHT

				if self.image_rect:
					# Do not rescale graph: keep graph coordinates and center them over image
					img_w, img_h = self.image.get_size()
					img_cx = self.image_rect.left + img_w / 2.0
					img_cy = self.image_rect.top + img_h / 2.0
					graph_cx = (min_x + max_x) / 2.0
					graph_cy = (min_y + max_y) / 2.0
					self.graph_scale = 1.0
					# translate graph so its center matches image center
					self.graph_offset_x = img_cx - graph_cx
					self.graph_offset_y = img_cy - graph_cy
				else:
					# no image: center in canvas without scaling
					self.graph_scale = 1.0
					canvas_center_x = SIDEBAR_W + canvas_w / 2.0
					canvas_center_y = canvas_h / 2.0
					graph_center_x = (min_x + max_x) / 2.0
					graph_center_y = (min_y + max_y) / 2.0
					self.graph_offset_x = canvas_center_x - graph_center_x
					self.graph_offset_y = canvas_center_y - graph_center_y
			else:
				# no spots
				self.graph_scale = 1.0
				self.graph_offset_x = 0
				self.graph_offset_y = 0

			self.start_spot = None
			self.end_spot = None
			self.astar = None
			self.algorithm_running = False
			self.algorithm_done = False

			# reset turns scroll when loading a new graph
			self.turns_scroll = 0
			self.text_path = []

			# snap spots to image roads for visual alignment
			if self.image is not None and self.graph is not None:
				self._snap_spots_to_image()
			return True
		except Exception as e:
			print('Error loading graph:', e)
			return False

	def _calculate_graph_offset(self):
		# kept for backwards compatibility; recompute simple centering
		if not self.graph or len(self.graph.spots) == 0:
			self.graph_offset_x = 0
			self.graph_offset_y = 0
			self.graph_scale = 1.0
			return
		min_x = min(s.position.x for s in self.graph.spots)
		max_x = max(s.position.x for s in self.graph.spots)
		min_y = min(s.position.y for s in self.graph.spots)
		max_y = max(s.position.y for s in self.graph.spots)

		canvas_w = SCREEN_WIDTH - SIDEBAR_W
		canvas_h = SCREEN_HEIGHT
		canvas_center_x = SIDEBAR_W + canvas_w / 2.0
		canvas_center_y = canvas_h / 2.0

		graph_center_x = (min_x + max_x) / 2.0
		graph_center_y = (min_y + max_y) / 2.0

		self.graph_scale = 1.0
		self.graph_offset_x = canvas_center_x - graph_center_x
		self.graph_offset_y = canvas_center_y - graph_center_y

	def get_spot_at(self, sx, sy, radius=14):
		if not self.graph:
			return None
		base_x = self.graph_offset_x + self.graph_extra_x
		base_y = self.graph_offset_y + self.graph_extra_y
		for i, s in enumerate(self.graph.spots):
			screen_x = int(s.position.x * self.graph_scale + base_x)
			screen_y = int(s.position.y * self.graph_scale + base_y)
			off = self.snap_offsets.get(i, (0, 0))
			screen_x += int(off[0])
			screen_y += int(off[1])
			if math.hypot(screen_x - sx, screen_y - sy) <= radius:
				return s
		return None

	def find_nearest_spot(self, sx, sy, max_radius=80):
		"""Return (spot, distance) of nearest spot within max_radius (canvas pixels), or (None, None)."""
		if not self.graph:
			return (None, None)
		best = None
		best_d2 = None
		base_x = self.graph_offset_x + self.graph_extra_x
		base_y = self.graph_offset_y + self.graph_extra_y
		for i, s in enumerate(self.graph.spots):
			x = float(s.position.x) * self.graph_scale + base_x
			y = float(s.position.y) * self.graph_scale + base_y
			off = self.snap_offsets.get(i, (0, 0))
			x += off[0]; y += off[1]
			d2 = (x - sx) ** 2 + (y - sy) ** 2
			if best_d2 is None or d2 < best_d2:
				best_d2 = d2
				best = s
		if best is None:
			return (None, None)
		d = math.sqrt(best_d2)
		if d <= max_radius:
			return (best, d)
		return (None, None)

	def _snap_spots_to_image(self, max_radius=40, threshold=100):
		"""Find nearest dark pixel in the loaded image for each spot and store a small screen-space offset.
		This is a visual-only adjustment so nodes appear on roads in the image.
		"""
		# get image array (width, height, 3)
		arr = pygame.surfarray.array3d(self.image)
		img_w, img_h = arr.shape[0], arr.shape[1]
		# compute grayscale brightness (shape (img_w, img_h))
		gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.uint8)
		# road mask: dark pixels
		road_mask = gray < threshold

		self.snap_offsets = {}
		# image top-left in canvas may be nudged by image_shift
		img_left = self.image_rect.left + getattr(self, 'image_shift_x', 0)
		img_top = self.image_rect.top + getattr(self, 'image_shift_y', 0)
		for i, s in enumerate(self.graph.spots):
			# base screen pos of spot
			bx = int(s.position.x * self.graph_scale + self.graph_offset_x)
			by = int(s.position.y * self.graph_scale + self.graph_offset_y)
			best = None
			best_d2 = None
			# search window in canvas coords
			x0 = max(img_left, bx - max_radius)
			x1 = min(img_left + img_w - 1, bx + max_radius)
			y0 = max(img_top, by - max_radius)
			y1 = min(img_top + img_h - 1, by + max_radius)
			# iterate neighborhood
			for px in range(x0, x1 + 1):
				for py in range(y0, y1 + 1):
					# map canvas pixel to image-local
					ix = px - img_left
					iy = py - img_top
					if ix < 0 or iy < 0 or ix >= img_w or iy >= img_h:
						continue
					if not road_mask[ix, iy]:
						continue
					dx = px - bx
					dy = py - by
					d2 = dx * dx + dy * dy
					if best_d2 is None or d2 < best_d2:
						best_d2 = d2
						best = (dx, dy)
				# store small offset if found
				if best is not None and best_d2 is not None and best_d2 <= (max_radius * max_radius):
					self.snap_offsets[i] = best
				else:
					# no road found nearby
					self.snap_offsets[i] = (0, 0)


	def waypoints2path(self, samples=6):
		"""Convert `self.astar.path` (list of spots) into a dense list of canvas points.
		`samples` specifies how many interpolated points between each waypoint (including endpoints).
		"""
		if not self.astar or not getattr(self.astar, 'path', None):
			return []
		pts = []
		nodes = self.astar.path
		for idx in range(len(nodes) - 1):
			a = nodes[idx]
			b = nodes[idx + 1]
			# find indices in graph to get snap offsets
			try:
				i_a = self.graph.spots.index(a)
			except ValueError:
				i_a = None
			try:
				i_b = self.graph.spots.index(b)
			except ValueError:
				i_b = None
			off_a = self.snap_offsets.get(i_a, (0, 0)) if i_a is not None else (0, 0)
			off_b = self.snap_offsets.get(i_b, (0, 0)) if i_b is not None else (0, 0)
			x1 = float(a.position.x) * self.graph_scale + self.graph_offset_x + self.graph_extra_x + off_a[0]
			y1 = float(a.position.y) * self.graph_scale + self.graph_offset_y + self.graph_extra_y + off_a[1]
			x2 = float(b.position.x) * self.graph_scale + self.graph_offset_x + self.graph_extra_x + off_b[0]
			y2 = float(b.position.y) * self.graph_scale + self.graph_offset_y + self.graph_extra_y + off_b[1]
			for s in range(samples):
				u = s / float(samples - 1)
				x = x1 * (1.0 - u) + x2 * u
				y = y1 * (1.0 - u) + y2 * u
				pts.append((int(round(x)), int(round(y))))
		# ensure last waypoint included
		last = nodes[-1]
		try:
			il = self.graph.spots.index(last)
		except ValueError:
			il = None
		off_l = self.snap_offsets.get(il, (0, 0)) if il is not None else (0, 0)
		xl = int(round(last.position.x * self.graph_scale + self.graph_offset_x + self.graph_extra_x + off_l[0]))
		yl = int(round(last.position.y * self.graph_scale + self.graph_offset_y + self.graph_extra_y + off_l[1]))
		if not pts or pts[-1] != (xl, yl):
			pts.append((xl, yl))
		return pts

	def start_astar(self):
		if self.start_spot and self.end_spot:
			# reset spot states and create AStar instance
			if hasattr(self.graph, 'clearSpots'):
				self.graph.clearSpots()
			self.astar = AStar(self.start_spot, self.end_spot)
			self.algorithm_running = True
			self.algorithm_done = False

	def step_astar(self):
		if self.astar and not self.algorithm_done:
			self.astar.update()
			if self.astar.isDone:
				self.algorithm_done = True
				self.algorithm_running = False

	def reset(self):
		if self.graph and hasattr(self.graph, 'clearSpots'):
			self.graph.clearSpots()
		self.astar = None
		self.start_spot = None
		self.end_spot = None
		self.algorithm_running = False
		self.algorithm_done = False
		self.turns_scroll = 0
		self.text_path = []
		self.chat_text = ""
		self.chat_caret = 0
		self.chat_view_start = 0
		self.chat_message = ""
		self.chat_message_timer = 0

	def is_crossroad(self, waypoint, path):
		"""Check if a waypoint is a crossroad (>2 neighbors)."""
		if waypoint < 0 or waypoint >= len(path):
			return False
		return len(path[waypoint].neighbors) > 2

	def is_uturn(self, waypoint, path):
		"""Check if waypoint is a U-turn based on distance to previous and next."""
		if len(path) == 0 or waypoint <= 0 or waypoint >= len(path) - 1:
			return False
		dist = math.sqrt((path[waypoint - 1].position.x - path[waypoint + 1].position.x) ** 2 +
		                  (path[waypoint - 1].position.y - path[waypoint + 1].position.y) ** 2)
		return dist < 30

	def get_turn_type(self, waypoint, path):
		"""Calculate turn direction at a crossroad waypoint: 0=right, 1=straight, 2=left, 3=uturn.
		Only call this on crossroad nodes (nodes with >2 neighbors).
		"""
		# only process crossroad nodes
		if not self.is_crossroad(waypoint, path):
			return 1  # default to straight for non-crossroads
		if waypoint <= 0 or waypoint >= len(path) - 1:
			return 1  # straight as default
		cur = path[waypoint]
		nxt = path[waypoint + 1]
		prev = path[waypoint - 1]
		# forward vector (from current to next)
		fx = nxt.position.x - cur.position.x
		fy = nxt.position.y - cur.position.y
		# incoming vector (from previous to current)
		nx = prev.position.x - cur.position.x
		ny = prev.position.y - cur.position.y
		# angle between forward and incoming vector
		ang = math.degrees(math.atan2(ny, nx) - math.atan2(fy, fx))
		ang = (ang + 180) % 360 - 180  # normalize to [-180, 180]
		self.last_ang = ang  # store for debugging
		# check for U-turn first
		if abs(ang) < 30:
			return 1
		elif ang > 0:
			if self.is_uturn(waypoint, path):
				return 3
			else:
				return 2
		else:
			if self.is_uturn(waypoint, path):
				return 3
			else:
				return 0

	def get_all_turns(self, path):
		"""Return list of turn types at all crossroads in path."""
		if not path:
			return []
		turns = []
		for i in range(len(path)):
			if self.is_crossroad(i, path):
				turn_type = self.get_turn_type(i, path)
				turns.append(turn_type)
		return turns

	def _chat_rect(self):
		return pygame.Rect(10, SCREEN_HEIGHT - 80, SIDEBAR_W - 20, 30)

	def _text_width(self, text):
		return font.size(text)[0]

	def _wrap_text(self, text, max_width):
		words = text.split()
		if not words:
			return [""]
		lines = []
		current = words[0]
		for word in words[1:]:
			candidate = current + " " + word
			if self._text_width(candidate) <= max_width:
				current = candidate
			else:
				lines.append(current)
				current = word
		lines.append(current)
		return lines

	def _prompt_caret_index(self, prompt):
		if prompt.startswith("Chat: "):
			return len("Chat: ") + self.chat_caret
		return len(prompt)

	def _fit_text_to_width(self, text, max_width):
		if self._text_width(text) <= max_width:
			self.chat_view_start = 0
			return text, 0
		start = min(self.chat_view_start, len(text))
		end = len(text)
		while self._text_width(text[start:end]) > max_width and end > start:
			end -= 1
		self.chat_view_start = start
		return text[start:end], start

	def _ensure_caret_visible(self):
		prompt = "Chat: " + self.chat_text
		caret_index = self._prompt_caret_index(prompt)
		max_width = self._chat_rect().width - 12
		if self._text_width(prompt) <= max_width:
			self.chat_view_start = 0
			return
		if caret_index < self.chat_view_start:
			self.chat_view_start = caret_index
		while self._text_width(prompt[self.chat_view_start:caret_index]) > max_width and self.chat_view_start < caret_index:
			self.chat_view_start += 1
		while self.chat_view_start > 0 and self._text_width(prompt[self.chat_view_start - 1:]) <= max_width:
			self.chat_view_start -= 1

	def _set_caret_from_click(self, x):
		chat_box = self._chat_rect()
		prompt = "Chat: " + self.chat_text
		rendered, start_idx = self._fit_text_to_width(prompt, chat_box.width - 12)
		rel_x = max(0, x - (chat_box.x + 6))
		caret_in_render = 0
		for i in range(len(rendered) + 1):
			if self._text_width(rendered[:i]) >= rel_x:
				caret_in_render = i
				break
		else:
			caret_in_render = len(rendered)
		caret_abs = start_idx + caret_in_render
		self.chat_caret = max(0, min(len(self.chat_text), caret_abs - len("Chat: ")))

	def _load_turns_from_file(self, filepath):
		try:
			with open(filepath, "rb") as f:
				return list(f.read())
		except Exception as e:
			self.chat_message = f"Error reading turns.bin: {e}"
			self.chat_message_timer = pygame.time.get_ticks()
			return []

	def _angle_between(self, ax, ay, bx, by):
		ang = math.degrees(math.atan2(ay, ax) - math.atan2(by, bx))
		return (ang + 180) % 360 - 180

	def _neighbors_sorted(self, spot):
		return sorted(spot.neighbors, key=lambda s: (s.position.x, s.position.y))

	def _is_uturn_distance(self, prev_spot, next_spot, threshold=30):
		if self.astar:
			return self.astar._distance(prev_spot, next_spot) < threshold
		dx = next_spot.position.x - prev_spot.position.x
		dy = next_spot.position.y - prev_spot.position.y
		return (dx * dx + dy * dy) ** 0.5 < threshold

	def _turn_code(self, prev_spot, curr_spot, next_spot):
		fx = next_spot.position.x - curr_spot.position.x
		fy = next_spot.position.y - curr_spot.position.y
		nx = prev_spot.position.x - curr_spot.position.x
		ny = prev_spot.position.y - curr_spot.position.y
		ang = self._angle_between(ny, nx, fy, fx)
		if abs(ang) < 30:
			return 1
		if ang > 0:
			return 3 if self._is_uturn_distance(prev_spot, next_spot) else 2
		return 0 if self._is_uturn_distance(prev_spot, next_spot) else 1

	def _pick_by_turn(self, prev_spot, curr_spot, candidates, turn_code):
		if prev_spot is None:
			return candidates[0] if candidates else None
		matching = [n for n in candidates if self._turn_code(prev_spot, curr_spot, n) == turn_code]
		if matching:
			return matching[0]
		fx = curr_spot.position.x - prev_spot.position.x
		fy = curr_spot.position.y - prev_spot.position.y
		best = None
		best_score = None
		for n in candidates:
			nx = n.position.x - curr_spot.position.x
			ny = n.position.y - curr_spot.position.y
			ang = self._angle_between(ny, nx, fy, fx)
			if turn_code == 1:
				score = abs(ang)
			elif turn_code == 2:
				score = -ang if ang > 0 else float("inf")
			elif turn_code == 0:
				score = ang if ang < 0 else float("inf")
			else:
				score = abs(abs(ang) - 180)
			if best is None or score < best_score:
				best = n
				best_score = score
		return best

	def _build_path_from_turns(self, turns_start_to_end):
		if not self.start_spot or not self.graph:
			return []

		def walk_from(first_neighbor):
			path = [self.start_spot, first_neighbor]
			prev = self.start_spot
			curr = first_neighbor
			turns = list(turns_start_to_end)
			steps = 0
			max_steps = len(self.graph.spots) * 4

			while steps < max_steps:
				steps += 1
				neighbors = [n for n in self._neighbors_sorted(curr) if not n.isWall]
				if len(neighbors) == 0:
					break

				if len(neighbors) == 2 and prev in neighbors:
					nxt = neighbors[0] if neighbors[1] is prev else neighbors[1]
					path.append(nxt)
					prev, curr = curr, nxt
					continue

				if len(neighbors) > 2 and turns:
					turn = turns.pop(0)
					candidates = neighbors if turn == 3 else ([n for n in neighbors if n is not prev] or neighbors)
					nxt = self._pick_by_turn(prev, curr, candidates, turn)
				else:
					if prev is None:
						nxt = neighbors[0]
					else:
						candidates = [n for n in neighbors if n is not prev]
						if not candidates:
							break
						nxt = self._pick_by_turn(prev, curr, candidates, 1)

				if nxt is None:
					break
				path.append(nxt)
				prev, curr = curr, nxt

				if not turns and len(curr.neighbors) <= 1:
					break

			# After consuming all turns, append one more neighbor if possible
			if not turns and len(path) >= 2:
				prev = path[-2]
				curr = path[-1]
				neighbors = [n for n in self._neighbors_sorted(curr) if not n.isWall]
				candidates = [n for n in neighbors if n is not prev]
				if not candidates and neighbors:
					candidates = neighbors
				if candidates:
					nxt = self._pick_by_turn(prev, curr, candidates, 1) or candidates[0]
					if nxt is not prev:
						path.append(nxt)

			return path, turns

		neighbors = [n for n in self._neighbors_sorted(self.start_spot) if not n.isWall]
		if not neighbors:
			return [self.start_spot]

		if not turns_start_to_end:
			path, _ = walk_from(neighbors[0])
		else:
			best = None
			for n in neighbors:
				candidate_path, remaining = walk_from(n)
				consumed = len(turns_start_to_end) - len(remaining)
				score = (consumed, len(candidate_path))
				if best is None or score > best[0]:
					best = (score, candidate_path)
			path = best[1] if best else [self.start_spot, neighbors[0]]

		return list(reversed(path))

	def _build_text_path_from_turns(self, filepath):
		turns_reversed = self._load_turns_from_file(filepath)
		turns_start_to_end = list(reversed(turns_reversed))
		return self._build_path_from_turns(turns_start_to_end)

	def _submit_chat(self):
		text = self.chat_text.strip()
		if not text:
			return
		if not self.start_spot:
			self.chat_message = "Select a START point first."
			self.chat_message_timer = pygame.time.get_ticks()
			return

		steps, clarification = parse_text_to_steps(text)
		if clarification:
			self.chat_message = clarification
			self.chat_message_timer = pygame.time.get_ticks()
			return

		try:
			write_turns_bin(steps, os.path.join(repo_root, "turns.bin"))
			self.chat_message = f"Wrote turns.bin: {steps}"
			self.chat_message_timer = pygame.time.get_ticks()
			self.astar = None
			self.end_spot = None
			self.algorithm_running = False
			self.algorithm_done = False
			self.text_path = self._build_text_path_from_turns(os.path.join(repo_root, "turns.bin"))
			self.chat_text = ""
			self.chat_caret = 0
			self.chat_view_start = 0
		except Exception as e:
			self.chat_message = f"Write failed: {e}"
			self.chat_message_timer = pygame.time.get_ticks()
			self.text_path = []

	def draw_sidebar(self):
		# background
		pygame.draw.rect(screen, LIGHT_GRAY, (0, 0, SIDEBAR_W, SCREEN_HEIGHT))
		y = 8
		screen.blit(title_font.render('Saved Maps', True, BLACK), (12, y))
		y += 36
		for i, name in enumerate(self.available):
			color = YELLOW if i == self.selected_idx else BLACK
			txt = font.render(name, True, color)
			screen.blit(txt, (16, y))
			y += 22

		# quick help (moved to top-right of canvas, so no controls here)

		# toggle button for showing/hiding the graph
		btn_w = SIDEBAR_W - 24
		btn_h = 28
		btn_x = 12
		btn_y = SCREEN_HEIGHT - 200
		self.toggle_button_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
		btn_color = GREEN if self.show_graph else GRAY
		pygame.draw.rect(screen, btn_color, self.toggle_button_rect)
		# button label
		label = 'Hide Graph' if self.show_graph else 'Show Graph'
		txt = font.render(label, True, BLACK)
		tx = btn_x + 8
		ty = btn_y + (btn_h - txt.get_height()) // 2
		screen.blit(txt, (tx, ty))

		# turn display section at bottom
		y = SCREEN_HEIGHT - 160
		pygame.draw.line(screen, BLACK, (12, y), (SIDEBAR_W - 12, y), 1)
		y += 8
		screen.blit(title_font.render('Turns', True, BLACK), (12, y))
		y += 28
		
		# show turns in reverse (most recent first) if path exists
		if self.astar and getattr(self.astar, 'path', None):
			path = self.astar.path
			turns = self.get_all_turns(path)
			if turns:
				# reverse turns list to show most recent first
				turns_rev = list(reversed(turns))
				turn_names = {0: 'Right', 1: 'Straight', 2: 'Left', 3: 'U-turn'}
				# show current (first in reversed) and next (second in reversed)
				for idx, turn_type in enumerate(turns_rev[:2]):
					label = turn_names.get(turn_type, '?')
					if idx == 0:
						title_txt = 'Current: ' + label
						color = GREEN
					else:
						title_txt = 'Next: ' + label
						color = BLUE
					screen.blit(font.render(title_txt, True, color), (12, y))
					y += 22
				# remaining turns (if any) show below in black, with scrolling
				remaining = turns_rev[2:]
				line_h = 20
				# available vertical space for remaining turns
				avail_h = (SCREEN_HEIGHT - y - 12)
				max_lines = max(0, avail_h // line_h)
				# clamp scroll
				if max_lines == 0:
					self.turns_scroll = 0
				else:
					max_scroll = max(0, len(remaining) - max_lines)
					self.turns_scroll = max(0, min(self.turns_scroll, max_scroll))
				start_idx = self.turns_scroll
				visible = remaining[start_idx:start_idx + max_lines]
				for t in visible:
					label = turn_names.get(t, '?')
					screen.blit(font.render(label, True, BLACK), (12, y))
					y += line_h
			else:
				screen.blit(font.render('No turns yet', True, BLACK), (12, y))
		else:
			screen.blit(font.render('Run A* to see turns', True, BLACK), (12, y))

		# Chat input area
		chat_box = self._chat_rect()
		pygame.draw.rect(screen, LIGHT_GRAY, chat_box)
		pygame.draw.rect(screen, DARK_GRAY, chat_box, 2 if self.chat_active else 1)
		prompt = "Chat: " + self.chat_text
		if not self.chat_text and not self.chat_active:
			prompt = "Chat: type route (requires 'start')"
		rendered, start_idx = self._fit_text_to_width(prompt, chat_box.width - 12)
		chat_text = font.render(rendered, True, BLACK)
		screen.blit(chat_text, (chat_box.x + 6, chat_box.y + 6))

		if self.chat_active and self.caret_visible:
			caret_x = chat_box.x + 6 + self._text_width(rendered[:self._prompt_caret_index(prompt) - start_idx])
			caret_y = chat_box.y + 6
			caret_h = font.get_height()
			pygame.draw.line(screen, BLACK, (caret_x, caret_y), (caret_x, caret_y + caret_h), 1)

		if self.chat_message:
			lines = self._wrap_text(self.chat_message, chat_box.width - 12)
			for i, line in enumerate(lines[:2]):
				msg = font.render(line, True, BLUE)
				screen.blit(msg, (chat_box.x + 6, chat_box.y - 18 - (len(lines[:2]) - 1 - i) * 16))

	def draw_canvas(self):
		# white background
		pygame.draw.rect(screen, WHITE, (SIDEBAR_W, 0, SCREEN_WIDTH - SIDEBAR_W, SCREEN_HEIGHT))

		# draw image if present
		if self.image and self.image_rect:
			img_x = self.image_rect.left + self.image_shift_x
			img_y = self.image_rect.top + self.image_shift_y
			screen.blit(self.image, (img_x, img_y))

		# draw graph
		# If we have a computed path, show only the path (as a smooth polyline) and start/end markers
		path = getattr(self.astar, 'path', None) if self.astar else None
		if path:
			pts = self.waypoints2path(samples=8)
			if pts and len(pts) > 1:
				pygame.draw.lines(screen, BLUE, False, pts, max(3, int(6 * self.graph_scale)))
		elif self.text_path:
			# draw text-driven path if available
			for i in range(len(self.text_path) - 1):
				a = self.text_path[i]
				b = self.text_path[i + 1]
				x1 = int(a.position.x * self.graph_scale + self.graph_offset_x + self.graph_extra_x)
				y1 = int(a.position.y * self.graph_scale + self.graph_offset_y + self.graph_extra_y)
				x2 = int(b.position.x * self.graph_scale + self.graph_offset_x + self.graph_extra_x)
				y2 = int(b.position.y * self.graph_scale + self.graph_offset_y + self.graph_extra_y)
				pygame.draw.line(screen, ORANGE, (x1, y1), (x2, y2), 3)
		else:
			# if show_graph True or A* is running, draw full graph; otherwise show only selected start/end markers
			if (self.show_graph or (self.astar and getattr(self.astar, 'openSet', None) is not None)) and self.graph:
				base_x = self.graph_offset_x + self.graph_extra_x
				base_y = self.graph_offset_y + self.graph_extra_y
				# edges
				for i, s in enumerate(self.graph.spots):
					for n in s.neighbors:
						x1 = int(s.position.x * self.graph_scale + base_x)
						y1 = int(s.position.y * self.graph_scale + base_y)
						o1 = self.snap_offsets.get(i, (0, 0))
						x1 += int(o1[0])
						y1 += int(o1[1])
						try:
							j = self.graph.spots.index(n)
						except ValueError:
							j = None
						x2 = int(n.position.x * self.graph_scale + base_x)
						y2 = int(n.position.y * self.graph_scale + base_y)
						if j is not None:
							o2 = self.snap_offsets.get(j, (0, 0))
							x2 += int(o2[0])
							y2 += int(o2[1])
						pygame.draw.line(screen, GRAY, (x1, y1), (x2, y2), 1)

				# spots
				for i, s in enumerate(self.graph.spots):
					x = int(s.position.x * self.graph_scale + base_x)
					y = int(s.position.y * self.graph_scale + base_y)
					off = self.snap_offsets.get(i, (0, 0))
					x += int(off[0])
					y += int(off[1])
					color = BLACK if getattr(s, 'isWall', False) else BLUE
					pygame.draw.circle(screen, color, (x, y), max(1, int(s.size * self.graph_scale)))

		# draw A* sets and path (only when graph visible and a path exists)
		if self.astar and self.show_graph and getattr(self.astar, 'path', None):
			# open set
			for s in getattr(self.astar, 'openSet', []):
				try:
					i = self.graph.spots.index(s)
				except ValueError:
					i = None
				x = int(s.position.x * self.graph_scale + self.graph_offset_x)
				y = int(s.position.y * self.graph_scale + self.graph_offset_y)
				if i is not None:
					off = self.snap_offsets.get(i, (0, 0))
					x += int(off[0])
					y += int(off[1])
				pygame.draw.circle(screen, GREEN, (x, y), max(1, int(s.size * self.graph_scale)) + 2, 1)
			# closed set
			for s in getattr(self.astar, 'closeSet', []):
				try:
					i = self.graph.spots.index(s)
				except ValueError:
					i = None
				x = int(s.position.x * self.graph_scale + self.graph_offset_x)
				y = int(s.position.y * self.graph_scale + self.graph_offset_y)
				if i is not None:
					off = self.snap_offsets.get(i, (0, 0))
					x += int(off[0])
					y += int(off[1])
				pygame.draw.circle(screen, RED, (x, y), max(1, int(s.size * self.graph_scale)) + 2, 1)
			# (path polyline is drawn earlier via waypoints2path)

		# draw start/end markers
		if self.start_spot:
			try:
				i = self.graph.spots.index(self.start_spot)
			except ValueError:
				i = None
			x = int(self.start_spot.position.x * self.graph_scale + self.graph_offset_x)
			y = int(self.start_spot.position.y * self.graph_scale + self.graph_offset_y)
			if i is not None:
				off = self.snap_offsets.get(i, (0, 0))
				x += int(off[0]); y += int(off[1])
			pygame.draw.circle(screen, GREEN, (x, y), max(1, int(self.start_spot.size * self.graph_scale)) + 4, 3)
		if self.end_spot:
			try:
				i = self.graph.spots.index(self.end_spot)
			except ValueError:
				i = None
			x = int(self.end_spot.position.x * self.graph_scale + self.graph_offset_x)
			y = int(self.end_spot.position.y * self.graph_scale + self.graph_offset_y)
			if i is not None:
				off = self.snap_offsets.get(i, (0, 0))
				x += int(off[0]); y += int(off[1])
			pygame.draw.circle(screen, RED, (x, y), max(1, int(self.end_spot.size * self.graph_scale)) + 4, 3)

		# draw path nodes as purple circles if a path exists (replace lines)
		if self.astar and getattr(self.astar, 'path', None):
			for idx, p in enumerate(self.astar.path):
				x = int(p.position.x * self.graph_scale + self.graph_offset_x + self.graph_extra_x)
				y = int(p.position.y * self.graph_scale + self.graph_offset_y + self.graph_extra_y)
				off = self.snap_offsets.get(idx, (0, 0))
				x += int(off[0]); y += int(off[1])
				pygame.draw.circle(screen, PURPLE, (x, y), max(2, int(4 * self.graph_scale)))

		# draw hover highlight (outline) if any
		if getattr(self, 'hover_spot', None) is not None:
			try:
				i = self.graph.spots.index(self.hover_spot)
			except Exception:
				i = None
			if i is not None:
				x = int(self.hover_spot.position.x * self.graph_scale + self.graph_offset_x + self.graph_extra_x)
				y = int(self.hover_spot.position.y * self.graph_scale + self.graph_offset_y + self.graph_extra_y)
				off = self.snap_offsets.get(i, (0, 0))
				x += int(off[0]); y += int(off[1])
				pygame.draw.circle(screen, YELLOW, (x, y), max(3, int(6 * self.graph_scale)), 2)

		# draw control hints at top-right of canvas
		y = 12
		hints = ['UP/DOWN: select map', 'ENTER: load', 'Click: pick spot', 'G: toggle graph', 'R: reset']
		for h in hints:
			txt = font.render(h, True, DARK_GRAY)
			screen.blit(txt, (SCREEN_WIDTH - 240, y))
			y += 18

	def run(self):
		running = True
		while running:
			clock.tick(FPS)
			for ev in pygame.event.get():
				if ev.type == pygame.QUIT:
					running = False
				elif ev.type == pygame.KEYDOWN:
					if self.chat_active:
						if ev.key == pygame.K_RETURN:
							self._submit_chat()
						elif ev.key == pygame.K_BACKSPACE:
							if self.chat_caret > 0:
								self.chat_text = self.chat_text[:self.chat_caret - 1] + self.chat_text[self.chat_caret:]
								self.chat_caret -= 1
								self._ensure_caret_visible()
						elif ev.key == pygame.K_LEFT:
							if self.chat_caret > 0:
								self.chat_caret -= 1
								self._ensure_caret_visible()
						elif ev.key == pygame.K_RIGHT:
							if self.chat_caret < len(self.chat_text):
								self.chat_caret += 1
								self._ensure_caret_visible()
						else:
							if ev.unicode and len(self.chat_text) < 120:
								self.chat_text = (
									self.chat_text[:self.chat_caret]
									+ ev.unicode
									+ self.chat_text[self.chat_caret:]
								)
								self.chat_caret += 1
								self._ensure_caret_visible()
						continue
					if ev.key == pygame.K_ESCAPE:
						running = False
					elif ev.key == pygame.K_UP:
						self.selected_idx = max(0, self.selected_idx - 1)
					elif ev.key == pygame.K_DOWN:
						self.selected_idx = min(len(self.available) - 1, self.selected_idx + 1)
					elif ev.key == pygame.K_RETURN:
						if self.available:
							name = self.available[self.selected_idx]
							self.load_graph(name)
					elif ev.key == pygame.K_SPACE:
						# if astar not started and start/end set, start
						if not self.astar and self.start_spot and self.end_spot:
							self.start_astar()
						elif self.astar and not self.algorithm_done:
							# step
							self.step_astar()
					elif ev.key == pygame.K_r:
						self.reset()
					elif ev.key == pygame.K_g:
						self.show_graph = not self.show_graph
					# nudge image: comma/period
					elif ev.key == pygame.K_COMMA:
						self.image_shift_x -= 10
					elif ev.key == pygame.K_PERIOD:
						self.image_shift_x += 10
					# nudge graph: [ and ] keys
					elif ev.key == pygame.K_LEFTBRACKET:
						self.graph_extra_x -= 10
					elif ev.key == pygame.K_RIGHTBRACKET:
						self.graph_extra_x += 10

				# mouse wheel events: pygame 2 uses MOUSEWHEEL, older versions use button 4/5
				elif ev.type == pygame.MOUSEWHEEL:
					mx, my = pygame.mouse.get_pos()
					if mx <= SIDEBAR_W:
						# ev.y: positive means scroll up, negative scroll down
						self.turns_scroll = max(0, self.turns_scroll - ev.y)
					continue
				elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (4, 5):
					# legacy wheel: 4=up, 5=down
					mx, my = ev.pos
					if mx <= SIDEBAR_W:
						if ev.button == 4:
							self.turns_scroll = max(0, self.turns_scroll - 1)
						else:
							self.turns_scroll = self.turns_scroll + 1
						# don't clamp here; draw_sidebar will clamp to valid range
					continue
				elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
					mx, my = ev.pos
					# check sidebar toggle button first
					if mx <= SIDEBAR_W and getattr(self, 'toggle_button_rect', None) and self.toggle_button_rect.collidepoint(mx, my):
						self.show_graph = not self.show_graph
						continue
					if self._chat_rect().collidepoint(mx, my):
						self.chat_active = True
						self._set_caret_from_click(mx)
						self._ensure_caret_visible()
						continue
					else:
						self.chat_active = False
					if mx > SIDEBAR_W:
						# canvas click: pick nearest spot within a reasonable radius
						spot, dist = self.find_nearest_spot(mx, my, max_radius=80)
						if spot:
							if not self.start_spot:
								self.start_spot = spot
							elif not self.end_spot and spot is not self.start_spot:
								self.end_spot = spot
								# immediately run full A* once two points are selected and show only the path
								self.start_astar()
								# run until done
								if self.astar:
									while not getattr(self.astar, 'isDone', False):
										self.astar.update()
									# after run, ensure we only display the path
									self.show_graph = False
					# print turn array to console
					path = getattr(self.astar, 'path', None)
					# if path:
					# 	turns = self.get_all_turns(path)
					# 	print(f"\n=== Path Generated ===")
					# 	print(f"Path length: {len(path)}")
					# 	print(f"Turn array: {turns}")
					# 	turn_names = {0: 'Right', 1: 'Straight', 2: 'Left', 3: 'U-turn'}
					# 	print("Turns (readable):")
					# 	for i, turn_type in enumerate(turns):
					# 		print(f"  Turn {i}: {turn_names.get(turn_type, '?')}")
					# 	print("=======================\n")
			screen.fill(WHITE)
			self.draw_sidebar()
			self.draw_canvas()
			# Clear transient chat message after 4 seconds
			if self.chat_message_timer:
				if pygame.time.get_ticks() - self.chat_message_timer > 4000:
					self.chat_message = ""
					self.chat_message_timer = 0

			if self.chat_active:
				now = pygame.time.get_ticks()
				if now - self.last_caret_toggle > self.caret_blink_ms:
					self.caret_visible = not self.caret_visible
					self.last_caret_toggle = now
			pygame.display.flip()

		pygame.quit()


if __name__ == '__main__':
	app = App()
	app.run()
