# New Playground Example

This example creates an interactive polyhedron editor using Bevy and egui.

## Features

- **Left Panel**: Operator stack with parameter controls
  - Select base shape (Platonic solids, prisms, antiprisms)
  - Add operators from dropdown menu
  - Enable/disable operators with checkboxes
  - Click operator name to show parameter sliders
  - Remove operators with ❌ button
  - Clear all operators button

- **Main Viewport**: 3D view of the polyhedron
  - Pan/orbit camera controls (drag to rotate, scroll to zoom)
  - Real-time updates when parameters change
  - PBR rendering with directional lighting

## Running

```bash
cargo run --example new_playground --features egui
```

## Usage

1. Select a base shape from the dropdown
2. Add operators by selecting from "Add Operator" dropdown
3. Click on an operator name to expand its parameters
4. Adjust ratio/height sliders to modify the effect
5. Toggle operators on/off with checkboxes
6. Remove operators with the ❌ button
7. The 3D view updates automatically

## Operator Parameters

- **Ratio**: Controls the proportion of the operation (0.0 to 1.0)
- **Height**: Controls extrusion/intrusion depth (-1.0 to 1.0)

Not all operators have both parameters - only relevant controls are shown.