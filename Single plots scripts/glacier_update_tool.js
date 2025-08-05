// Function to create glacier mask
function getGlacierMask(ndsi, ndwi, ndsiThresh, ndwiThresh) {
  // Create glacier mask using thresholds
  var glacierMask = ndsi.gt(ndsiThresh)     // NDSI > ndsiThresh for glacier detection
    .and(ndwi.lt(ndwiThresh));             // NDWI < ndwiThresh to exclude water bodies

  // Apply morphological operations to clean up
  glacierMask = glacierMask
    .focal_min({radius: 1, units: 'pixels'})
    .focal_max({radius: 2, units: 'pixels'});

  return glacierMask;
}

// Load the RGI v7 dataset
var rgi = ee.FeatureCollection("projects/sat-io/open-datasets/RGI/RGI_VECTOR_MERGED_V7");

// Define minimum glacier size (in km²)
var MIN_GLACIER_SIZE = 3;  // Default value, can be changed via UI

// Define threshold constants as mutable variables
var ndsiThreshold = 0.85;  // Default NDSI threshold for glacier detection
var ndwiThreshold = 0.3;   // Default NDWI threshold to exclude water bodies

// Default to Western Canada initially
var currentRegion = '02';
var currentSubregion = null;
var filterNamedGlaciersOnly = false;
var currentRegionGlaciers = rgi.filter(ee.Filter.eq('o1region', currentRegion));
var currentRegionFiltered = currentRegionGlaciers
  .filter(ee.Filter.gte('area_km2', MIN_GLACIER_SIZE));

// Initial display
Map.centerObject(currentRegionFiltered, 5);
Map.addLayer(currentRegionFiltered, {color: 'blue'}, 
  'RGI Glaciers (>= ' + MIN_GLACIER_SIZE + ' km²)');

print('=== REGION ' + currentRegion + ' (WESTERN CANADA AND USA) TOTALS ===');
print('All glaciers:', currentRegionGlaciers.size());
print('Glaciers >= ' + MIN_GLACIER_SIZE + ' km²:', currentRegionFiltered.size());

// Create left-side control panel for glacier mask controls
var controlPanel = ui.Panel({
  layout: ui.Panel.Layout.flow('vertical'),
  style: {
    width: '350px',
    padding: '8px',
    backgroundColor: 'white',
    border: '1px solid #ccc'
  }
});

// Add header label to control panel
controlPanel.add(ui.Label({
  value: '🏔️ Glacier Update Tool',
  style: {fontSize: '20px', fontWeight: 'bold', margin: '10px 0', color: '#2E7D32'}
}));

// Collapsible Info Panel
var infoPanel = ui.Panel({
  style: { shown: false, margin: '0 -8px' } // Hide by default
});

var infoButton = ui.Button({
  label: 'ℹ️ Show App Info',
  onClick: function() {
    var shown = infoPanel.style().get('shown');
    infoPanel.style().set('shown', !shown);
    infoButton.setLabel(shown ? 'ℹ️ Show App Info' : '🔼 Hide App Info');
  },
  style: { stretch: 'horizontal', margin: '0 0 10px 0' }
});

controlPanel.add(infoButton);
controlPanel.add(infoPanel);

// Add detailed app description to the collapsible panel
infoPanel.add(ui.Label({
  value: '📋 ABOUT THIS TOOL',
  style: {fontSize: '14px', fontWeight: 'bold', margin: '15px 8px 5px 8px', color: '#1976D2'}
}));

infoPanel.add(ui.Label({
  value: 'This application updates glacier outlines from the Randolph Glacier Inventory version 7.0 (RGI v7) using current Sentinel-2 satellite imagery to track glacier changes over time.',
  style: {fontSize: '11px', color: '444', margin: '0 8px 10px 8px', whiteSpace: 'pre-wrap'}
}));

infoPanel.add(ui.Label({
  value: '📊 DATA SOURCES:',
  style: {fontSize: '12px', fontWeight: 'bold', margin: '10px 8px 5px 8px'}
}));

infoPanel.add(ui.Label({
  value: '• Baseline: RGI v7.0 (2023 release)\n' +
         '• Imagery: Sentinel-2 L2A (10m resolution)\n' +
         '• Period: August only (minimal snow)\n' +
         '• Cloud filter: <10% coverage',
  style: {fontSize: '11px', color: '555', margin: '0 8px 10px 8px', whiteSpace: 'pre'}
}));

infoPanel.add(ui.Label({
  value: '🎯 KEY FEATURES:',
  style: {fontSize: '12px', fontWeight: 'bold', margin: '10px 8px 5px 8px'}
}));

infoPanel.add(ui.Label({
  value: '• NDSI threshold for snow/ice detection\n' +
         '• NDWI threshold to exclude water bodies\n' +
         '• Real-time threshold adjustment\n' +
         '• Automatic area change calculation\n' +
         '• Multi-year composite analysis\n' +
         '• Export to Shapefile or EE Asset',
  style: {fontSize: '11px', color: '555', margin: '0 8px 10px 8px', whiteSpace: 'pre'}
}));

infoPanel.add(ui.Label({
  value: '📈 WORKFLOW:',
  style: {fontSize: '12px', fontWeight: 'bold', margin: '10px 8px 5px 8px'}
}));

infoPanel.add(ui.Label({
  value: '1. Select RGI region/subregion below\n' +
         '2. Draw polygon on map (right panel)\n' +
         '3. Click "Apply Polygon Selection"\n' +
         '4. Process with Sentinel-2 imagery\n' +
         '5. Adjust thresholds if needed\n' +
         '6. Export updated outlines',
  style: {fontSize: '11px', color: '555', margin: '0 8px 15px 8px', whiteSpace: 'pre'}
}));

// Add a separator line
controlPanel.add(ui.Panel({
  widgets: [],
  style: {
    height: '1px',
    backgroundColor: '#E0E0E0',
    margin: '10px 0'
  }
}));

controlPanel.add(ui.Label({
  value: '⚙️ DETECTION PARAMETERS',
  style: {fontSize: '14px', fontWeight: 'bold', margin: '10px 0', color: '#1976D2'}
}));

// NDSI threshold slider
controlPanel.add(ui.Label({
  value: 'NDSI Threshold:',
  style: {fontWeight: 'bold', margin: '10px 0 5px 0'}
}));

var ndsiValueLabel = ui.Label({
  value: ndsiThreshold.toFixed(2),
  style: {margin: '0 0 5px 0'}
});
controlPanel.add(ndsiValueLabel);

var ndsiSlider = ui.Slider({
  min: 0,
  max: 1,
  value: ndsiThreshold,
  step: 0.01,
  style: {stretch: 'horizontal'}
});
var lastGlacierLayer = null;
var ndsi = null;
var ndwi = null;

// Simple throttling variables
var lastNdsiUpdate = 0;
var lastNdwiUpdate = 0;
var UPDATE_THROTTLE = 300; // milliseconds

ndsiSlider.onChange(function(value) {
  ndsiThreshold = value;
  ndsiValueLabel.setValue(value.toFixed(2));

  // Simple throttling using Date.now()
  var now = Date.now();
  if (now - lastNdsiUpdate < UPDATE_THROTTLE) {
    return;
  }
  lastNdsiUpdate = now;

  // Only update glacier mask if ndsi and ndwi are available
  if (ndsi && ndwi) {
    if (lastGlacierLayer) {
      Map.layers().remove(lastGlacierLayer);
    }

    var newGlacierMask = getGlacierMask(ndsi, ndwi, ndsiThreshold, ndwiThreshold);
    lastGlacierLayer = ui.Map.Layer(newGlacierMask, {palette: ['cyan']}, 'Glacier Mask');
    Map.layers().insert(Map.layers().length() - 1, lastGlacierLayer);
  } else {
    print('NDSI threshold updated to ' + value.toFixed(2) + '. Process imagery first to see glacier mask updates.');
  }
});
controlPanel.add(ndsiSlider);

// RGI Region and Subregion Selectors
controlPanel.add(ui.Label({
  value: 'RGI Region Selection:',
  style: {fontWeight: 'bold', margin: '15px 0 5px 0'}
}));

// Region Selector
var rgiRegions = {
  '01: Alaska': '01',
  '02: Western Canada and USA': '02',
  '03: Arctic Canada North': '03',
  '04: Arctic Canada South': '04',
  '05: Greenland Periphery': '05',
  '06: Iceland': '06',
  '07: Svalbard and Jan Mayen': '07',
  '08: Scandinavia': '08',
  '09: Russian Arctic': '09',
  '10: North Asia': '10',
  '11: Central Europe': '11',
  '12: Caucasus and Middle East': '12',
  '13: Central Asia': '13',
  '14: South Asia West': '14',
  '15: South Asia East': '15',
  '16: Low Latitudes': '16',
  '17: Southern Andes': '17',
  '18: New Zealand': '18',
  '19: Subantarctic and Antarctic Islands': '19',
  '20: Antarctic Mainland': '20'
};

// Subregions for each region
var rgiSubregions = {
  '01': {
    '01-01: Brooks Range': '01-01',
    '01-02: Alaska Range': '01-02',
    '01-03: Aleutian Islands': '01-03',
    '01-04: Chugach Mountains': '01-04',
    '01-05: Coast Mountains': '01-05'
  },
  '02': {
    '02-01: Alaska Range': '02-01',
    '02-02: Coast Mountains': '02-02',
    '02-03: Canadian Rocky Mountains': '02-03',
    '02-04: Cascade Range and Sierra Nevada': '02-04',
    '02-05: Southern Rocky Mountains': '02-05'
  }
  // Add other regions and subregions as needed
};

var regionSelect = ui.Select({
  items: Object.keys(rgiRegions),
  placeholder: 'Select a region',
  onChange: function(key) {
    var regionId = rgiRegions[key];
    currentRegion = regionId;
    updateGlaciers();
    updateSubregionSelector(regionId);
  }
});
controlPanel.add(regionSelect);

// Subregion Selector (initially empty)
var subregionSelect = ui.Select({
  placeholder: 'Select a subregion',
  onChange: function(key) {
    currentSubregion = rgiSubregions[currentRegion][key];
    updateGlaciers();
  }
});
controlPanel.add(subregionSelect);

// Checkbox to filter for named glaciers
var namedGlacierCheckbox = ui.Checkbox({
  label: 'Show named glaciers only',
  value: false,
  onChange: function(checked) {
    filterNamedGlaciersOnly = checked;
    updateGlaciers();
  }
});
controlPanel.add(namedGlacierCheckbox);

// Add minimum size filter input
controlPanel.add(ui.Label({
  value: 'Minimum Glacier Size (km²):',
  style: {fontWeight: 'bold', margin: '10px 0 5px 0'}
}));

var sizeFilterPanel = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {margin: '0 0 10px 0'}
});

var minSizeTextbox = ui.Textbox({
  value: String(MIN_GLACIER_SIZE),
  style: {width: '80px'}
});

var applySizeButton = ui.Button({
  label: 'Apply',
  onClick: function() {
    var newSize = parseFloat(minSizeTextbox.getValue());
    if (!isNaN(newSize) && newSize >= 0) {
      MIN_GLACIER_SIZE = newSize;
      updateGlaciers();
      print('Minimum glacier size updated to: ' + MIN_GLACIER_SIZE + ' km²');
    } else {
      alert('Please enter a valid number >= 0');
    }
  },
  style: {width: '60px'}
});

sizeFilterPanel.add(minSizeTextbox);
sizeFilterPanel.add(applySizeButton);
controlPanel.add(sizeFilterPanel);

// Add helper text
controlPanel.add(ui.Label({
  value: 'Enter 0 to show all glaciers',
  style: {fontSize: '10px', color: '888', margin: '0 0 10px 0'}
}));

// NDWI threshold slider
controlPanel.add(ui.Label({
  value: 'NDWI Threshold:',
  style: {fontWeight: 'bold', margin: '15px 0 5px 0'}
}));

var ndwiValueLabel = ui.Label({
  value: ndwiThreshold.toFixed(2),
  style: {margin: '0 0 5px 0'}
});
controlPanel.add(ndwiValueLabel);

var ndwiSlider = ui.Slider({
  min: -1,
  max: 1,
  value: ndwiThreshold,
  step: 0.01,
  style: {stretch: 'horizontal'}
});
ndwiSlider.onChange(function(value) {
  ndwiThreshold = value;
  ndwiValueLabel.setValue(value.toFixed(2));

  // Simple throttling using Date.now()
  var now = Date.now();
  if (now - lastNdwiUpdate < UPDATE_THROTTLE) {
    return;
  }
  lastNdwiUpdate = now;

  // Only update glacier mask if ndsi and ndwi are available
  if (ndsi && ndwi) {
    if (lastGlacierLayer) {
      Map.layers().remove(lastGlacierLayer);
    }

    var newGlacierMask = getGlacierMask(ndsi, ndwi, ndsiThreshold, ndwiThreshold);
    lastGlacierLayer = ui.Map.Layer(newGlacierMask, {palette: ['cyan']}, 'Glacier Mask');
    Map.layers().insert(Map.layers().length() - 1, lastGlacierLayer);
  } else {
    print('NDWI threshold updated to ' + value.toFixed(2) + '. Process imagery first to see glacier mask updates.');
  }
});
controlPanel.add(ndwiSlider);

function updateSubregionSelector(regionId) {
  var subregions = rgiSubregions[regionId] || {};
  subregionSelect.items().reset(Object.keys(subregions));
  subregionSelect.setPlaceholder('Select a subregion');
  currentSubregion = null; // Reset subregion when region changes
}

function updateGlaciers() {
  var glaciers = rgi.filter(ee.Filter.eq('o1region', currentRegion));
  if (currentSubregion) {
    glaciers = glaciers.filter(ee.Filter.eq('o2region', currentSubregion));
  }
  
  if (filterNamedGlaciersOnly) {
    // Filter for glaciers where glac_name is not null AND not empty string
    glaciers = glaciers
      .filter(ee.Filter.notNull(['glac_name']))
      .filter(ee.Filter.neq('glac_name', ''));
  }
  currentRegionGlaciers = glaciers;
  
  currentRegionFiltered = currentRegionGlaciers
    .filter(ee.Filter.gte('area_km2', MIN_GLACIER_SIZE));
  
  Map.layers().reset();
  Map.addLayer(currentRegionFiltered, {color: 'blue'}, 
    'RGI Glaciers (>= ' + MIN_GLACIER_SIZE + ' km²)');
  
  var regionName = '';
  var regionKeys = Object.keys(rgiRegions);
  for (var i = 0; i < regionKeys.length; i++) {
    var key = regionKeys[i];
    if (rgiRegions[key] === currentRegion) {
      regionName = key;
      break;
    }
  }

  print('=== REGION ' + currentRegion + ' (' + regionName + ') TOTALS ===');
  print('All glaciers:', currentRegionGlaciers.size());
  print('Glaciers >= ' + MIN_GLACIER_SIZE + ' km²:', currentRegionFiltered.size());
}

// Add the control panel to the root on the left side
ui.root.insert(0, controlPanel);

// Create a panel for controls
var panel = ui.Panel({
  style: {width: '400px', position: 'top-right'}
});
Map.add(panel);

panel.add(ui.Label({
  value: 'Glacier Update Tool',
  style: {fontSize: '18px', fontWeight: 'bold'}
}));

// Global variable to store selected glaciers
var selectedGlaciers = null;

// Create drawing tools
var drawingTools = Map.drawingTools();
drawingTools.setShown(true);

function setupDrawingLayer() {
  while (drawingTools.layers().length() > 0) {
    drawingTools.layers().remove(drawingTools.layers().get(0));
  }
  var geometryLayer = ui.Map.GeometryLayer({
    geometries: null,
    name: 'geometry',
    color: 'red'
  });
  drawingTools.layers().add(geometryLayer);
}

setupDrawingLayer();

// Function to update glacier outlines with Sentinel-2
function updateGlacierOutlines() {
  if (!selectedGlaciers) {
    alert('Please select glaciers first using the polygon tool');
    return;
  }
  
  print('\n=== UPDATING GLACIER OUTLINES WITH SENTINEL-2 (AUGUST ONLY) ===');
  
  // Get the bounding box of selected glaciers with buffer
  var glacierBounds = selectedGlaciers.geometry().bounds();
  var bufferDistance = 1000; // 1km buffer around glaciers
  var studyArea = glacierBounds.buffer(bufferDistance);
  
  // Get years from UI
  var startYear = parseInt(startYearBox.getValue()) || 2023;
  var endYear = parseInt(endYearBox.getValue()) || 2024;
  
  print('Searching for August imagery from ' + startYear + ' to ' + endYear);
  
  // Load Sentinel-2 for August only across multiple years
  var s2Collection = ee.ImageCollection([]);
  
  for (var year = startYear; year <= endYear; year++) {
    var augustImages = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(studyArea)
      .filterDate(year + '-08-01', year + '-08-31')  // August only
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)); // Stricter cloud filter
    
    s2Collection = s2Collection.merge(augustImages);
  }
  
  print('August Sentinel-2 images found:', s2Collection.size());
  
  // Cloud masking function
  function maskS2clouds(image) {
    var qa = image.select('QA60');
    var cloudBitMask = 1 << 10;
    var cirrusBitMask = 1 << 11;
    var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
        .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
    return image.updateMask(mask)
        .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
        .copyProperties(image, ['system:time_start']);
  }
  
  // Apply cloud masking
  var s2Masked = s2Collection.map(maskS2clouds);
  
  // Create median composite from all August images
  var composite = s2Masked.median().clip(studyArea);
  
  // Calculate NDSI (Normalized Difference Snow Index)
  ndsi = composite.normalizedDifference(['B3', 'B11']).rename('NDSI');
  
  // Calculate NDWI (Normalized Difference Water Index) to filter out water
  ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');
  
  print('NDSI and NDWI calculated. Sliders are now active for real-time updates.');
  
  // Create glacier mask using the encapsulated function
  var glacierMask = getGlacierMask(ndsi, ndwi, ndsiThreshold, ndwiThreshold);
  
  // Mask to areas near RGI glaciers only (with buffer)
  var rgiBuffer = selectedGlaciers.geometry().buffer(500); // 500m buffer
  glacierMask = glacierMask.clip(rgiBuffer);
  
  // Vectorize the glacier mask
  var vectors = glacierMask.selfMask().reduceToVectors({
    geometry: rgiBuffer,
    scale: 10,
    geometryType: 'polygon',
    eightConnected: false,
    maxPixels: 1e8,
    bestEffort: true
  });
  
  // Filter out small polygons (noise)
  var minArea = 10000; // 10,000 m² = 0.01 km²
  var glacierOutlines = vectors.filter(ee.Filter.gt('count', minArea / 100));
  
  // Add original RGI IDs to new outlines based on overlap - WITH PROPER ERROR HANDLING
  var updatedOutlines = glacierOutlines.map(function(feature) {
    var geom = feature.geometry();
    
    // Simplify geometry to avoid errors
    var simplifiedGeom = geom.simplify(10); // 10m simplification
    
    // Calculate area with error margin
    var newArea = simplifiedGeom.area(1); // 1m error margin
    
    // Find which RGI glacier this outline overlaps with most
    var overlappingGlaciers = selectedGlaciers.filterBounds(simplifiedGeom);
    var hasOverlap = overlappingGlaciers.size().gt(0);
    
    // Create base properties that always exist
    var baseProperties = ee.Dictionary({
      'update_date': ee.Date(Date.now()).format('YYYY-MM-dd'),
      'source': 'Sentinel-2_August',
      'years_used': startYear + '-' + endYear,
      'method': 'NDSI_threshold',
      'new_area_km2': newArea.divide(1e6)
    });
    
    // Conditionally add RGI properties if there's an overlap
    var properties = ee.Algorithms.If(
      hasOverlap,
      // If there's an overlap, add RGI properties
      baseProperties.combine(ee.Dictionary({
        'rgi_id': overlappingGlaciers.first().get('rgi_id'),
        'glac_name': overlappingGlaciers.first().get('glac_name'),
        'o1region': overlappingGlaciers.first().get('o1region'),
        'o2region': overlappingGlaciers.first().get('o2region'),
        'original_area_km2': overlappingGlaciers.first().get('area_km2'),
        'area_change_km2': newArea.divide(1e6).subtract(
          ee.Number(overlappingGlaciers.first().get('area_km2'))
        )
      })),
      // If no overlap, use base properties with null values
      baseProperties.combine(ee.Dictionary({
        'rgi_id': 'no_match',
        'glac_name': null,
        'o1region': null,
        'o2region': null,
        'original_area_km2': null,
        'area_change_km2': null
      }))
    );
    
    return ee.Feature(simplifiedGeom, properties);
  });
  
  // Remove any features that might have invalid geometries
  updatedOutlines = updatedOutlines.filter(ee.Filter.notNull(['new_area_km2']));
  
  // Count features with and without RGI matches
  var matchedCount = updatedOutlines.filter(ee.Filter.neq('rgi_id', 'no_match')).size();
  var unmatchedCount = updatedOutlines.filter(ee.Filter.eq('rgi_id', 'no_match')).size();
  
  print('Total updated outlines:', updatedOutlines.size());
  print('Outlines matched to RGI:', matchedCount);
  print('Outlines without RGI match:', unmatchedCount);
  
  // Calculate statistics - handle potential errors
  updatedOutlines.size().evaluate(function(count) {
    if (count > 0) {
      // Calculate area only for matched glaciers
      var matchedOutlines = updatedOutlines.filter(ee.Filter.neq('rgi_id', 'no_match'));
      
      matchedOutlines.geometry().area(1).divide(1e6).evaluate(function(newArea) {
        if (newArea !== null) {
          print('Updated total area:', newArea.toFixed(2), 'km²');
          
          selectedGlaciers.aggregate_sum('area_km2').evaluate(function(oldArea) {
            if (oldArea !== null && newArea !== null) {
              var change = ((newArea - oldArea) / oldArea * 100).toFixed(1);
              print('Area change:', change + '%');
            }
          });
        }
      });
    } else {
      print('No glacier outlines detected in the imagery');
    }
  });
  
  // Visualize results with improved color palette
  Map.addLayer(composite.select(['B4', 'B3', 'B2']).divide(10000), 
    {min: 0, max: 0.3}, 'August Composite RGB', false);
  Map.addLayer(ndsi, {min: -0.5, max: 0.8, palette: ['#8B4513', '#F5F5DC', '#4169E1']}, 
    'NDSI (August)', false);
  Map.addLayer(ndwi, {min: -0.5, max: 0.8, palette: ['#DC143C', '#F0F8FF', '#0000FF']}, 
    'NDWI (August)', false);
  Map.addLayer(glacierMask.selfMask(), {palette: ['#00FFFF']}, 
    'Glacier Mask', false);
  Map.addLayer(selectedGlaciers, {color: '#FF0000', width: 3}, 'Original RGI Outlines');
  Map.addLayer(updatedOutlines, {color: '#FFD700', width: 2}, 'Updated Outlines (August)');
  
  // Color code matched vs unmatched outlines with better contrast
  var matchedVis = updatedOutlines.filter(ee.Filter.neq('rgi_id', 'no_match'));
  var unmatchedVis = updatedOutlines.filter(ee.Filter.eq('rgi_id', 'no_match'));
  Map.addLayer(matchedVis, {color: '#32CD32', width: 2}, 'Matched to RGI', false);
  Map.addLayer(unmatchedVis, {color: '#FF8C00', width: 2}, 'No RGI Match', false);
  
  // Enable export of updated outlines
  exportUpdatedButton.setDisabled(false);
  exportUpdatedButton.onClick(function() {
    Export.table.toDrive({
      collection: updatedOutlines,
      description: 'Updated_Glacier_Outlines_August_S2_' + startYear + '_' + endYear,
      fileFormat: 'SHP',
      folder: 'GEE_Glacier_Updates'
    });
    alert('Export of updated outlines started! Check the Tasks tab.');
  });
  
  // Enable export of glacier mask as EE asset
  exportAssetButton.setDisabled(false);
  exportAssetButton.onClick(function() {
    // Create a final glacier mask with current threshold values
    var finalGlacierMask = getGlacierMask(ndsi, ndwi, ndsiThreshold, ndwiThreshold)
      .clip(selectedGlaciers.geometry().buffer(500))
      .selfMask(); // Only export pixels classified as glacier
    
    // Add metadata to the image
    var maskWithMetadata = finalGlacierMask
      .set('ndsi_threshold', ndsiThreshold)
      .set('ndwi_threshold', ndwiThreshold)
      .set('creation_date', ee.Date(Date.now()).format('YYYY-MM-dd'))
      .set('source', 'Sentinel-2_August_' + startYear + '_' + endYear)
      .set('method', 'NDSI_NDWI_thresholding')
      .set('scale_meters', 10)
      .set('description', 'Glacier mask derived from Sentinel-2 August imagery using NDSI and NDWI thresholds');
    
    // Helper function to pad numbers (GEE-compatible)
    function pad(num, size) {
      var s = '000000000' + num;
      return s.substr(s.length - size);
    }
    
    // Generate a unique asset name with timestamp (GEE-compatible format)
    var now = new Date();
    var timestamp = now.getFullYear() + '_' + 
                   pad(now.getMonth() + 1, 2) + '_' + 
                   pad(now.getDate(), 2) + '_' + 
                   pad(now.getHours(), 2) + 
                   pad(now.getMinutes(), 2);
    var assetName = 'Glacier_Mask_August_S2_' + startYear + '_' + endYear + '_' + timestamp;
    
    Export.image.toAsset({
      image: maskWithMetadata,
      description: assetName,
      assetId: assetName, // Simplified - will be saved to your default asset folder
      scale: 10,
      region: selectedGlaciers.geometry().buffer(500),
      maxPixels: 1e9,
      crs: 'EPSG:4326'
    });
    
    alert('Export of glacier mask as EE asset started!\n' +
          'Asset name: ' + assetName + '\n' +
          'NDSI threshold: ' + ndsiThreshold.toFixed(2) + '\n' +
          'NDWI threshold: ' + ndwiThreshold.toFixed(2) + '\n\n' +
          'Check the Tasks tab to monitor progress.');
    
    print('Glacier mask export details:');
    print('Asset name:', assetName);
    print('NDSI threshold:', ndsiThreshold);
    print('NDWI threshold:', ndwiThreshold);
    print('Scale: 10 meters');
  });
  
  return updatedOutlines;
}

// Function to filter glaciers within drawn polygon
function filterByDrawnPolygon() {
  var layer = drawingTools.layers().get(0);
  var geometries = layer.geometries();
  
  if (geometries.length() > 0) {
    var geometryList = [];
    for (var i = 0; i < geometries.length(); i++) {
      geometryList.push(geometries.get(i));
    }
    
    var polygon = ee.Geometry.MultiPolygon(geometryList);
    
    print('\n=== POLYGON SELECTION APPLIED ===');
    
    // Filter glaciers
selectedGlaciers = currentRegionFiltered.filterBounds(polygon);
    
    // Update map
    Map.layers().reset();
Map.addLayer(currentRegionFiltered, {color: 'lightblue'}, 
      'All Glaciers (background)', false);
    Map.addLayer(polygon, {color: 'red', fillColor: '00000020'}, 'Selected Area');
    Map.addLayer(selectedGlaciers, {color: 'blue'}, 
      'Selected RGI Glaciers');
    
    selectedGlaciers.size().evaluate(function(count) {
      if (count > 0) {
        selectedGlaciers.aggregate_sum('area_km2').evaluate(function(area) {
          resultsLabel.setValue('Selected ' + count + ' glaciers\n' +
            'Total area: ' + area.toFixed(2) + ' km²\n\n' +
            'Click "Update with Sentinel-2" to create new outlines');
          
          updateButton.setDisabled(false);
          exportRGIButton.setDisabled(false);
        });
      } else {
        resultsLabel.setValue('No glaciers found in selected area.');
        updateButton.setDisabled(true);
      }
    });
  }
}

// UI Elements
panel.add(ui.Label({
  value: '1. Draw polygon to select glaciers:',
  style: {fontWeight: 'bold', margin: '10px 0 5px 0'}
}));

var applyButton = ui.Button({
  label: 'Apply Polygon Selection',
  onClick: filterByDrawnPolygon,
  style: {stretch: 'horizontal'}
});
panel.add(applyButton);

var clearButton = ui.Button({
  label: 'Clear Drawing',
  onClick: function() {
    setupDrawingLayer();
    Map.layers().reset();
Map.addLayer(currentRegionFiltered, {color: 'blue'}, 
      'RGI Glaciers (>= ' + MIN_GLACIER_SIZE + ' km²)');
    resultsLabel.setValue('Draw a polygon to select glaciers');
    updateButton.setDisabled(true);
    exportRGIButton.setDisabled(true);
    exportUpdatedButton.setDisabled(true);
    exportAssetButton.setDisabled(true);
    selectedGlaciers = null;
  },
  style: {stretch: 'horizontal'}
});
panel.add(clearButton);

var resultsLabel = ui.Label({
  value: 'Draw a polygon to select glaciers',
  style: {margin: '10px 0', whiteSpace: 'pre'}
});
panel.add(resultsLabel);

panel.add(ui.Label({
  value: '2. Update glacier outlines:',
  style: {fontWeight: 'bold', margin: '10px 0 5px 0'}
}));

// Add year selectors
panel.add(ui.Label('August imagery years:', {margin: '10px 0 5px 0'}));
var startYearBox = ui.Textbox({
  placeholder: 'Start year',
  value: '2023',
  style: {stretch: 'horizontal'}
});
panel.add(startYearBox);

var endYearBox = ui.Textbox({
  placeholder: 'End year',
  value: '2024',
  style: {stretch: 'horizontal'}
});
panel.add(endYearBox);

var updateButton = ui.Button({
  label: 'Update with Sentinel-2 (August only)',
  onClick: updateGlacierOutlines,
  style: {stretch: 'horizontal'},
  disabled: true
});
panel.add(updateButton);

panel.add(ui.Label({
  value: '3. Export results:',
  style: {fontWeight: 'bold', margin: '10px 0 5px 0'}
}));

var exportRGIButton = ui.Button({
  label: 'Export Original RGI Selection',
  onClick: function() {
    Export.table.toDrive({
      collection: selectedGlaciers,
      description: 'RGI_Selected_Glaciers',
      fileFormat: 'SHP',
      folder: 'GEE_Glacier_Updates'
    });
    alert('Export started! Check the Tasks tab.');
  },
  style: {stretch: 'horizontal'},
  disabled: true
});
panel.add(exportRGIButton);

var exportUpdatedButton = ui.Button({
  label: 'Export Updated Outlines to Drive',
  style: {stretch: 'horizontal'},
  disabled: true
});
panel.add(exportUpdatedButton);

var exportAssetButton = ui.Button({
  label: 'Export Glacier Mask as EE Asset',
  style: {stretch: 'horizontal'},
  disabled: true
});
panel.add(exportAssetButton);

// Help text
panel.add(ui.Label({
  value: '\nAugust-only analysis:\n' +
    '• Minimal seasonal snow\n' +
    '• Cloud threshold: 10%\n' +
    '• NDSI > 0.85 for glacier detection\n' +
    '• NDWI < 0.3 to exclude water bodies\n' +
    '• Handles glaciers without RGI match\n' +
    '• Green = matched, Orange = no match',
  style: {fontSize: '12px', color: '666', whiteSpace: 'pre'}
}));

