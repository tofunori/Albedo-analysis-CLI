# MODIS Edge Effect Analysis - Complete Reference

## What is Edge Effect?

Edge effect measures how surrounded each glacier pixel is by other ice pixels using a **1-9 proximity score** based on Moore neighborhood analysis.

## Calculation Method

```
Moore Neighborhood (3×3 grid):
[N][N][N]
[N][X][N]  → Count ice neighbors around target pixel X
[N][N][N]

Edge Effect Score = Number of ice neighbors + 1
Range: 1 (isolated) to 9 (completely surrounded)
```

## All Possible Configurations

### Edge Effect 1 (0 neighbors)
```
[ ][ ][ ]
[ ][X][ ]  ← Completely isolated ice pixel
[ ][ ][ ]
```
**Characteristics:** Isolated glacier remnant, highest contamination risk

### Edge Effect 2 (1 neighbor)
```
[ ][ ][ ]    [ ][ ][ ]    [I][ ][ ]
[I][X][ ]    [ ][X][I]    [X][ ][ ]
[ ][ ][ ]    [ ][ ][ ]    [ ][ ][ ]
```
**Characteristics:** Very edge pixels, high contamination from non-ice surfaces

### Edge Effect 3 (2 neighbors)
```
[I][ ][ ]    [ ][I][ ]    [I][I][ ]
[I][X][ ]    [I][X][ ]    [X][ ][ ]
[ ][ ][ ]    [ ][ ][ ]    [ ][ ][ ]
```
**Characteristics:** Edge pixels with some connectivity, moderate contamination

### Edge Effect 4 (3 neighbors)
```
[I][I][ ]    [I][ ][ ]    [ ][I][ ]
[I][X][ ]    [I][X][I]    [I][X][ ]
[ ][ ][ ]    [I][ ][ ]    [I][ ][ ]
```
**Characteristics:** Corner or edge positions, transitional quality

### Edge Effect 5 (4 neighbors)
```
[ ][I][ ]    [I][I][ ]    [I][ ][ ]
[I][X][I]    [I][X][ ]    [I][X][I]
[ ][I][ ]    [ ][I][ ]    [ ][I][ ]
```
**Characteristics:** Cross patterns, moderate interior positions

### Edge Effect 6 (5 neighbors)
```
[I][I][ ]    [ ][I][I]    [I][I][I]
[I][X][I]    [I][X][I]    [X][ ][ ]
[ ][I][ ]    [ ][I][ ]    [I][ ][ ]
```
**Characteristics:** Good connectivity, reduced edge effects

### Edge Effect 7 (6 neighbors)
```
[I][I][I]    [I][I][ ]    [ ][I][I]
[I][X][ ]    [I][X][I]    [I][X][I]
[I][I][ ]    [I][I][I]    [I][I][I]
```
**Characteristics:** Near-interior positions, good data quality

### Edge Effect 8 (7 neighbors)
```
[I][I][I]    [I][I][I]
[I][X][ ]    [ ][X][I]
[I][I][I]    [I][I][I]
```
**Characteristics:** Almost completely surrounded, excellent quality

### Edge Effect 9 (8 neighbors)
```
[I][I][I]
[I][X][I]  ← Completely surrounded ice pixel
[I][I][I]
```
**Characteristics:** Perfect interior position, maximum data quality

## Data Quality Interpretation

| Score | Quality | Description | Typical Use |
|-------|---------|-------------|-------------|
| 1-2 | 🔴 **Poor** | High contamination risk | Exclude from analysis |
| 3-4 | 🟠 **Low** | Edge effects present | Use with caution |
| 5-6 | 🟡 **Moderate** | Transitional zones | Good for trend analysis |
| 7-8 | 🟢 **Good** | Interior positions | Preferred for analysis |
| 9 | 🟢 **Excellent** | Perfect interior | Ideal reference pixels |

## Real-World Glacier Patterns

### Small Glaciers/Ice Caps
- **Common:** Edge effects 1-5
- **Rare:** Edge effects 7-9
- **Characteristics:** High perimeter-to-area ratio

### Large Ice Sheets
- **Common:** Edge effects 6-9
- **Rare:** Edge effects 1-3
- **Characteristics:** Extensive interior coverage

### Fragmented Ice
- **Common:** Edge effects 1-4
- **Pattern:** Scattered isolated pixels
- **Challenge:** High contamination

## Applications

### Quality Filtering
```
High Quality: Edge Effect ≥ 6
Medium Quality: Edge Effect 4-5
Low Quality: Edge Effect ≤ 3
```

### Trend Analysis
- Use interior pixels (7-9) for climate signal
- Avoid edge pixels (1-3) for contamination
- Monitor edge migration over time

### Regional Studies
- Compare edge effect distributions
- Identify glacier fragmentation
- Assess data reliability by region