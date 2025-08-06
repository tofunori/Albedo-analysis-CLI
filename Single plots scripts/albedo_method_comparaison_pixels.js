// Standalone MOD09GA/MYD09GA Processing Script - Ren et al. 2021 Pipeline with Glacier Masking, RenOriginal QA, and Multi-Product Comparison Pixel Export
// This script processes daily MODIS images using the Ren et al. 2021 albedo retrieval pipeline for MOD09GA (Terra) and MYD09GA (Aqua).
// Includes MOD10A1/MYD10A1 (Terra/Aqua) and MCD43A3 (combined) for comparison, with their respective processing and QA.
// For MOD09GA/MYD09GA, uses 'renoriginal' QA mode: minimal cloud mask with -999 fill, topography correction, BRDF correction,
// snow/ice classification, broadband albedo computation, and glacier masking.
// For MOD10A1/MYD10A1 and MCD43A3, uses their standard QA and processing.
// Then, it samples pixels daily with coordinates, albedo, glacier fraction, reflectances (for MOD09GA/MYD09GA), NDSI, topographic variables, etc.,
// and exports a combined CSV with 'qa_mode' and 'method' columns to distinguish.
// Satellite overpass times (local solar time):
// - Terra (MOD products): ~10:30 AM (descending node)
// - Aqua (MYD products): ~1:30 PM (ascending node)
// - MCD43A3: 16-day composite from both Terra and Aqua
// QA Mode Descriptions:
// - MOD09GA/MYD09GA: 'renoriginal' - Minimal cloud mask (bits 0,1) with -999 fill for masked pixels
// - MOD10A1/MYD10A1: 'standard_qa' - Basic QA level 'good', excludes inland water, cloud, NDSI/visible/temp/height fails, SWIR anomaly, high solar zenith
// - MCD43A3: 'qa_0_and_1' - Accepts both full BRDF inversions (QA=0) and magnitude inversions (QA=1)
// Glacier Fraction Column:
// - Values range from 0 to 1 (0-100% glacier coverage within the MODIS pixel)
// - Only pixels with glacier_fraction > 0.50 are included in the export (due to glacier masking)
// - Calculated by aggregating 30m glacier outline to 500m MODIS resolution

// ==============================================================================
// CONFIGURATION SECTION - ALL USER-MODIFIABLE SETTINGS
// ==============================================================================

// ------------------------------------------------------------------------------
// 1. GLACIER SELECTION
// ------------------------------------------------------------------------------
// Single glacier mode (default) - set to false for multi-glacier processing
var SINGLE_GLACIER_MODE = true;
var GLACIER_ASSET = 'projects/tofunori/assets/colombia_icefield_image';

// Multi-glacier option (set SINGLE_GLACIER_MODE = false to use)
var GLACIER_ASSETS = [
  'projects/tofunori/assets/haig_glacier_image',
  'projects/tofunori/assets/athabasca_glacier_image',
  'projects/tofunori/assets/coropuna_glacier_image',
  'projects/tofunori/assets/colombia_icefield_image'
];

// ------------------------------------------------------------------------------
// 2. TEMPORAL SETTINGS
// ------------------------------------------------------------------------------
var START_DATE = '2002-01-01';
var END_DATE = '2025-01-01';
var START_MONTH = 5;  // May
var END_MONTH = 10;   // October

// ------------------------------------------------------------------------------
// 3. GLACIER PROCESSING SETTINGS
// ------------------------------------------------------------------------------
var GLACIER_CONFIG = {
  scale: 30, // Resolution for glacier outline rasterization (meters)
  abundance_threshold: 0.50, // Minimum glacier fraction per MODIS pixel (50%)
  modis_scale: 500 // MODIS pixel resolution (meters)
};

// ------------------------------------------------------------------------------
// 4. METHOD SELECTION
// ------------------------------------------------------------------------------
var METHODS = ['MOD09GA', 'MYD09GA', 'mod10a1', 'myd10a1', 'mcd43a3']; // All methods enabled for comparison

// ------------------------------------------------------------------------------
// 5. QUALITY ASSURANCE SETTINGS
// ------------------------------------------------------------------------------

// MOD09GA/MYD09GA QA Settings
var MOD09GA_QA_CONFIG = {
  mode: 'renoriginal', // QA mode for MOD09GA/MYD09GA
  cloud_bit_0: true,   // Use cloud bit 0 for masking
  cloud_bit_1: true,   // Use cloud bit 1 for masking
  fill_value: -999     // Fill value for masked pixels
};

// MOD10A1/MYD10A1 QA Settings  
// Inlined MODIS Collections from config.js
var MODIS_COLLECTIONS = {
  MOD10A1: 'MODIS/061/MOD10A1', // Terra
  MYD10A1: 'MODIS/061/MYD10A1', // Aqua
  MCD43A3: 'MODIS/061/MCD43A3' // Combined Terra+Aqua
};
// Albedo Coefficients (Defined here)
var ICE_COEFFICIENTS = {b1: 0.160, b2: 0.291, b3: 0.243, b4: 0.116, b5: 0.112, b7: 0.081, constant: -0.0015};
var SNOW_COEFFICIENTS = {b1: 0.1574, b2: 0.2789, b3: 0.3829, b5: 0.1131, b7: 0.0694, constant: -0.0093};
// BRDF Coefficients (Defined here)
var SNOW_BRDF_COEFFICIENTS = {
  b1: { c1: 0.00083, c2: 0.00384, c3: 0.00452, theta_c: 0.34527 },
  b2: { c1: 0.00123, c2: 0.00459, c3: 0.00521, theta_c: 0.34834 },
  b3: { c1: 0.00000, c2: 0.00001, c3: 0.00002, theta_c: 0.12131 },
  b5: { c1: 0.00663, c2: 0.01081, c3: 0.01076, theta_c: 0.46132 },
  b7: { c1: 0.00622, c2: 0.01410, c3: 0.01314, theta_c: 0.55261 }
};
var ICE_BRDF_COEFFICIENTS = {
  b1: { c1: -0.00054, c2: 0.00002, c3: 0.00001, theta_c: 0.17600 },
  b2: { c1: -0.00924, c2: 0.00033, c3: -0.00005, theta_c: 0.31750 },
  b3: { c1: -0.00369, c2: 0.00000, c3: 0.00007, theta_c: 0.27632 },
  b4: { c1: -0.02920, c2: -0.00810, c3: 0.00462, theta_c: 0.52360 },
  b5: { c1: -0.02388, c2: 0.00656, c3: 0.00227, theta_c: 0.58473 },
  b7: { c1: -0.02081, c2: 0.00683, c3: 0.00390, theta_c: 0.57500 }
};
var MOD10A1_QA_CONFIG = {
  STANDARD: {
    basicLevel: 'good',
    excludeInlandWater: true,
    excludeVisibleScreenFail: true,
    excludeNDSIScreenFail: true,
    excludeTempHeightFail: true,
    excludeSWIRAnomaly: true,
    excludeProbablyCloudy: true,
    excludeProbablyClear: false,
    excludeHighSolarZenith: true
  },
  BIT_MAPPING: [
    {flag: 'excludeInlandWater', bit: 0, mask: 1},
    {flag: 'excludeVisibleScreenFail', bit: 1, mask: 2},
    {flag: 'excludeNDSIScreenFail', bit: 2, mask: 4},
    {flag: 'excludeTempHeightFail', bit: 3, mask: 8},
    {flag: 'excludeSWIRAnomaly', bit: 4, mask: 16},
    {flag: 'excludeProbablyCloudy', bit: 5, mask: 32},
    {flag: 'excludeProbablyClear', bit: 6, mask: 64},
    {flag: 'excludeHighSolarZenith', bit: 7, mask: 128}
  ]
};

// ------------------------------------------------------------------------------
// 6. EXPORT SETTINGS
// ------------------------------------------------------------------------------
var EXPORT_CONFIG = {
  folder: 'MODIS_Pixel_Analysis',  // Google Drive folder for exports
  formats: {
    long: true,      // Export long format (original structure)
    pivoted: true,   // Export pivoted format (wide structure)
    summary: true    // Export daily summary
  }
};

// ------------------------------------------------------------------------------
// 7. DEBUG SETTINGS
// ------------------------------------------------------------------------------
var DEBUG_MODE = false;  // Set to true to enable detailed debug prints (may cause memory issues)

// ==============================================================================
// END OF CONFIGURATION SECTION
// ==============================================================================

// Function to extract glacier name from asset path
function extractGlacierName(assetPath) {
  var parts = assetPath.split('/');
  var filename = parts[parts.length - 1];
  var cleanName = filename.replace('_glacier_image', '')
                          .replace('_icefield_image', '')
                          .replace('_image', '');
  // Capitalize each word
  return cleanName.split('_')
                  .map(function(word) { 
                    return word.charAt(0).toUpperCase() + word.slice(1); 
                  })
                  .join('_');
}

// MODIS Collections
var MODIS_COLLECTION = 'MODIS/061/MOD09GA'; // Primary for MOD09GA method (Terra)
var AQUA_COLLECTION = 'MODIS/061/MYD09GA'; // Primary for MYD09GA method (Aqua)

// Band configurations
var REFL_BANDS = ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07'];
var TOPO_BANDS_ALL = ['sur_refl_b01_topo', 'sur_refl_b02_topo', 'sur_refl_b03_topo', 'sur_refl_b04_topo', 'sur_refl_b05_topo', 'sur_refl_b07_topo'];
var TOPO_BANDS_SNOW = ['sur_refl_b01_topo', 'sur_refl_b02_topo', 'sur_refl_b03_topo', 'sur_refl_b05_topo', 'sur_refl_b07_topo'];
var BAND_NUMS_ALL = ['b1', 'b2', 'b3', 'b4', 'b5', 'b7'];
var BAND_NUMS_SNOW = ['b1', 'b2', 'b3', 'b5', 'b7'];

function getMOD10A1BasicQAMask(img, level) {
  var basic = img.select('NDSI_Snow_Cover_Basic_QA');
  var mask;
  switch(level){
    case 'best': mask = basic.eq(0); break;
    case 'good': mask = basic.lte(1); break;
    case 'ok': mask = basic.lte(2); break;
    case 'all': mask = basic.lte(3); break;
    default: mask = basic.lte(1);
  }
  return mask.and(basic.neq(211)).and(basic.neq(239));
}
function getMOD10A1AlgorithmFlagsMask(img, flags){
  var alg = img.select('NDSI_Snow_Cover_Algorithm_Flags_QA').uint8();
  var mask = ee.Image(1);
  MOD10A1_QA_CONFIG.BIT_MAPPING.forEach(function(m){
    if(flags[m.flag]){
      mask = mask.and(alg.bitwiseAnd(m.mask).eq(0));
    }
  });
  return mask;
}
function createMOD10A1ComprehensiveMask(img, preset){
  var basicMask = getMOD10A1BasicQAMask(img, preset.basicLevel||'good');
  var flagsMask = getMOD10A1AlgorithmFlagsMask(img, preset);
  return basicMask.and(flagsMask);
}
function createMOD10A1StandardQualityMask(img) {
  return createMOD10A1ComprehensiveMask(img, MOD10A1_QA_CONFIG.STANDARD);
}
// Generic process function for MOD10A1/MYD10A1 (Terra/Aqua)
function processMODIS10A1(image, glacierOutlines, satellite) {
  // Using standard QA: basicLevel='good', excludes clouds, water, and other issues
  var qualityMask = createMOD10A1StandardQualityMask(image);
  var filtered = image.updateMask(qualityMask);
 
  var finalAlbedo = filtered
      .select('Snow_Albedo_Daily_Tile')
      .multiply(0.01) // Convert 0-100 to 0-1
      .rename('broadband_albedo_' + satellite);
 
  var ndsiData = image
      .select('NDSI')
      .multiply(0.0001) // Convert 0-10000 to 0-1
      .rename('NDSI');
 
  var glacierMask = createGlacierMask(glacierOutlines).reproject(finalAlbedo.projection());
 
  var maskedAlbedo = finalAlbedo.updateMask(glacierMask)
                     .rename('broadband_albedo_' + satellite + '_masked');
  return filtered
           .addBands(finalAlbedo)
           .addBands(maskedAlbedo)
           .addBands(ndsiData)
           .copyProperties(image, ['system:time_start'])
           .set('method', satellite).set('qa_mode', 'standard_qa');
}
// MCD43A3 QA Settings
var MCD43A3_QA_CONFIG = {
  ACCEPT_QA_0_AND_1: true,
  MANDATORY_QA_BANDS: [
    'BRDF_Albedo_Band_Mandatory_Quality_shortwave',
    'BRDF_Albedo_Band_Mandatory_Quality_vis',
    'BRDF_Albedo_Band_Mandatory_Quality_nir'
  ],
  QUALITY_FLAGS: {
    FULL_INVERSION: 0,
    MAGNITUDE_INVERSION: 1
  }
};
function createMCD43A3QualityMask(image) {
  var shortQA = image.select('BRDF_Albedo_Band_Mandatory_Quality_shortwave');
  var visQA = image.select('BRDF_Albedo_Band_Mandatory_Quality_vis');
  var nirQA = image.select('BRDF_Albedo_Band_Mandatory_Quality_nir');
 
  var shortGood = shortQA.bitwiseAnd(3).lte(1);
  var visGood = visQA.bitwiseAnd(3).lte(1);
  var nirGood = nirQA.bitwiseAnd(3).lte(1);
 
  var goodCount = shortGood.add(visGood).add(nirGood);
  return shortGood.and(goodCount.gte(2));
}
// Inlined processMCD43A3 from mcd43a3.js
function processMCD43A3(image, glacierOutlines) {
  // Using QA that accepts both full inversion (0) and magnitude inversion (1)
  var qualityMask = createMCD43A3QualityMask(image);
  var filteredImage = image.updateMask(qualityMask);
 
  var blackSkySW = filteredImage.select('Albedo_BSA_shortwave').multiply(0.001);
 
  var whiteSkySW = filteredImage.select('Albedo_WSA_shortwave').multiply(0.001);
 
  var broadbandAlbedo = blackSkySW.rename('broadband_albedo_mcd43a3');
 
  var glacierMask = createGlacierMask(glacierOutlines).reproject(broadbandAlbedo.projection());
  var maskedAlbedo = broadbandAlbedo.updateMask(glacierMask)
                      .rename('broadband_albedo_mcd43a3_masked');
  return filteredImage
           .addBands(broadbandAlbedo)
           .addBands(maskedAlbedo)
           .addBands(blackSkySW.rename('black_sky_albedo_mcd43a3'))
           .addBands(whiteSkySW.rename('white_sky_albedo_mcd43a3'))
           .copyProperties(image, ['system:time_start'])
           .set('method', 'mcd43a3').set('qa_mode', 'qa_0_and_1');
}
// Topographic Data (From original)
var demCollection = ee.ImageCollection('JAXA/ALOS/AW3D30/V4_1');
var dem = demCollection.select('DSM').mosaic().setDefaultProjection(demCollection.first().select('DSM').projection());
var slope = ee.Terrain.slope(dem);
var aspect = ee.Terrain.aspect(dem);
// Glacier Initialization (From original)
function initializeGlacierData(assetPath) {
  var glacierImage = ee.Image(assetPath || GLACIER_ASSET);
  var glacierBounds = glacierImage.geometry().bounds(1);
  var glacierOutlines = glacierImage.gt(0).selfMask().reduceToVectors({
    geometry: glacierBounds,
    scale: GLACIER_CONFIG.scale,
    geometryType: 'polygon'
  });
  return {
    image: glacierImage,
    bounds: glacierBounds,
    outlines: glacierOutlines,
    geometry: glacierOutlines.geometry(),
    assetPath: assetPath || GLACIER_ASSET
  };
}
// Glacier Masking (From original)
// This function creates a mask that selects MODIS pixels with >50% glacier coverage
// It works by:
// 1. Creating a high-resolution (30m) glacier map from the glacier outlines
// 2. Aggregating to MODIS resolution (500m) to calculate glacier fraction per pixel
// 3. Selecting only pixels where glacier fraction > 0.50 (50%)
function createGlacierMask(glacierOutlines) {
  var glacierBounds = glacierOutlines.geometry().bounds(1);
  var glacierMap = ee.Image(0).paint(glacierOutlines, 1).unmask(0)
    .clip(glacierBounds)
    .setDefaultProjection({crs: 'EPSG:4326', scale: GLACIER_CONFIG.scale});
 
  var glacierFraction = glacierMap
    .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1000})
    .reproject({crs: 'EPSG:4326', scale: GLACIER_CONFIG.modis_scale});
 
  var mask50 = glacierFraction.gt(GLACIER_CONFIG.abundance_threshold);
  var glacierBoundsMask = ee.Image(0).paint(glacierOutlines, 1).gt(0);
 
  return mask50.and(glacierBoundsMask);
}
// New function to get glacier fraction (not just mask)
function getGlacierFraction(glacierOutlines) {
  var glacierBounds = glacierOutlines.geometry().bounds(1);
  var glacierMap = ee.Image(0).paint(glacierOutlines, 1).unmask(0)
    .clip(glacierBounds)
    .setDefaultProjection({crs: 'EPSG:4326', scale: GLACIER_CONFIG.scale});
 
  var glacierFraction = glacierMap
    .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1000})
    .reproject({crs: 'EPSG:4326', scale: GLACIER_CONFIG.modis_scale})
    .rename('glacier_fraction');
 
  return glacierFraction;
}
// QA Filtering Functions for MOD09GA/MYD09GA (using config)
function qualityFilter(image) {
  var qa = image.select("state_1km");
  var mask = ee.Image(1);
  
  if (MOD09GA_QA_CONFIG.cloud_bit_0) {
    var cloud0BitMask = (1 << 0);
    mask = mask.and(qa.bitwiseAnd(cloud0BitMask).eq(0));
  }
  
  if (MOD09GA_QA_CONFIG.cloud_bit_1) {
    var cloud1BitMask = (1 << 1);
    mask = mask.and(qa.bitwiseAnd(cloud1BitMask).eq(0));
  }
  
  return image.updateMask(mask).unmask(MOD09GA_QA_CONFIG.fill_value);
}
// Topography Correction (From original, for MOD09GA/MYD09GA)
function topographyCorrection(image) {
  var deg2rad = Math.PI / 180;
  var szRad = image.select('SolarZenith').multiply(0.01 * deg2rad);
  var saRad = image.select('SolarAzimuth').multiply(0.01 * deg2rad);
  var vzRad = image.select('SensorZenith').multiply(0.01 * deg2rad);
  var vaRad = image.select('SensorAzimuth').multiply(0.01 * deg2rad);
 
  var imgGeom = image.geometry();
  var slopeRad = slope.clip(imgGeom).multiply(deg2rad);
  var aspectRad = aspect.clip(imgGeom).multiply(deg2rad);
 
  var cosV = slopeRad.cos().multiply(vzRad.cos())
      .add(slopeRad.sin().multiply(vzRad.sin())
      .multiply(aspectRad.subtract(vaRad).cos()));
  var cosS = slopeRad.cos().multiply(szRad.cos())
      .add(slopeRad.sin().multiply(szRad.sin())
      .multiply(aspectRad.subtract(saRad).cos()));
 
  var scaledBands = REFL_BANDS.map(function(b) {
    return image.select(b).multiply(0.0001).rename(b + '_topo');
  });
 
  return image
    .addBands(ee.Image.cat(scaledBands))
    .addBands(cosV.acos().multiply(180/Math.PI).rename('SensorZenith_corrected'))
    .addBands(cosS.acos().multiply(180/Math.PI).rename('SolarZenith_corrected'))
    .addBands([cosS.lte(0).rename('shadow_mask'),
               image.select('sur_refl_b01').eq(32767).rename('sat_vis')]);
}
// Snow/Ice Classification (From original, for MOD09GA/MYD09GA)
function classifySnowIce(image) {
  var green = image.select('sur_refl_b04_topo');
  var swir = image.select('sur_refl_b06_topo');
  var ndsiData = green.subtract(swir).divide(green.add(swir)).rename('NDSI');
 
  return image.addBands([ndsiData, ndsiData.gt(0.4).rename('snow_mask')]);
}
// BRDF Anisotropic Correction (From original, for MOD09GA/MYD09GA)
function applyBRDFAnisotropicCorrection(image, surfaceType) {
  var sensorZenithCorrected = image.select('SensorZenith_corrected').multiply(Math.PI / 180);
  var solarZenithCorrected = image.select('SolarZenith_corrected').multiply(Math.PI / 180);
  var relativeAzimuth = image.select('SensorAzimuth').multiply(0.01 * Math.PI / 180)
                             .subtract(image.select('SolarAzimuth').multiply(0.01 * Math.PI / 180));
  var coeffTable = surfaceType === 'snow' ? SNOW_BRDF_COEFFICIENTS : ICE_BRDF_COEFFICIENTS;
  var bandsList = surfaceType === 'snow' ? TOPO_BANDS_SNOW : TOPO_BANDS_ALL;
  var bandNums = surfaceType === 'snow' ? BAND_NUMS_SNOW : BAND_NUMS_ALL;
  var narrowbands = bandsList.map(function (band, idx) {
    var bandNum = bandNums[idx];
    if (surfaceType === 'ice' && bandNum === 'b7') {
      return image.select(band).rename('narrowband_' + bandNum);
    }
   
    var coeff = coeffTable[bandNum];
    if (!coeff) return null;
    var g1, g2, g3;
    if (surfaceType === 'snow') {
      var theta2 = sensorZenithCorrected.multiply(sensorZenithCorrected);
      g1 = theta2;
      g2 = theta2.multiply(relativeAzimuth.cos());
      g3 = g2.multiply(relativeAzimuth.cos());
      var anisotropy = theta2.add(0.5).subtract(Math.PI * Math.PI / 8).multiply(coeff.c1)
        .add(g2.multiply(coeff.c2))
        .add(g3.add(0.25).subtract(Math.PI * Math.PI / 16).multiply(coeff.c3))
        .multiply(solarZenithCorrected.divide(coeff.theta_c).exp());
    } else {
      var cosTheta = sensorZenithCorrected.cos();
      var theta2 = sensorZenithCorrected.multiply(sensorZenithCorrected);
      g1 = cosTheta;
      g2 = theta2.multiply(relativeAzimuth.cos());
      g3 = g2.multiply(relativeAzimuth.cos());
      var anisotropy = cosTheta.subtract(2 / 3).multiply(coeff.c1)
        .add(g2.multiply(coeff.c2))
        .add(g3.add(0.25).subtract(Math.PI * Math.PI / 16).multiply(coeff.c3))
        .multiply(solarZenithCorrected.divide(coeff.theta_c).exp());
    }
    return image.select(band).subtract(anisotropy).min(0.99).rename('narrowband_' + bandNums[idx]);
  }).filter(function (img) { return img !== null; });
  return image.addBands(ee.Image.cat(narrowbands));
}
// Broadband Albedo Computation (From original, for MOD09GA/MYD09GA)
function computeBroadbandAlbedo(image, satellite) {
  var b1 = image.select('narrowband_b1');
  var b2 = image.select('narrowband_b2');
  var b3 = image.select('narrowband_b3');
  var b4 = image.bandNames().contains('narrowband_b4') ? image.select('narrowband_b4') : ee.Image.constant(0);
  var b5 = image.select('narrowband_b5');
  var b7 = image.select('narrowband_b7');
  var ice = b1.multiply(ICE_COEFFICIENTS.b1).add(b2.multiply(ICE_COEFFICIENTS.b2)).add(b3.multiply(ICE_COEFFICIENTS.b3))
    .add(b4.multiply(ICE_COEFFICIENTS.b4)).add(b5.multiply(ICE_COEFFICIENTS.b5)).add(b7.multiply(ICE_COEFFICIENTS.b7))
    .add(ICE_COEFFICIENTS.constant).rename('ice_albedo');
  var snow = b1.multiply(SNOW_COEFFICIENTS.b1).add(b2.multiply(SNOW_COEFFICIENTS.b2)).add(b3.multiply(SNOW_COEFFICIENTS.b3))
    .add(b5.multiply(SNOW_COEFFICIENTS.b5)).add(b7.multiply(SNOW_COEFFICIENTS.b7))
    .add(SNOW_COEFFICIENTS.constant).rename('snow_albedo');
  var broadband = ice.where(image.select('snow_mask'), snow).rename('broadband_albedo_' + satellite);
  return image.addBands([ice, snow, broadband]);
}
// Main Processing Function for MOD09GA/MYD09GA (From original)
function processRenMethod(image, glacierOutlines, satellite) {
  var roi = glacierOutlines.geometry();
  image = image.clip(roi);
  var filtered = qualityFilter(image);
  var topoImg = topographyCorrection(filtered);
 
  var validMask = ee.Image(1);
  REFL_BANDS.forEach(function(b) {
    validMask = validMask.and(topoImg.select(b + '_topo').gte(0));
  });
  topoImg = topoImg.updateMask(validMask);
 
  var classified = classifySnowIce(topoImg);
 
  var nbSnow = applyBRDFAnisotropicCorrection(classified, 'snow');
  var nbIce = applyBRDFAnisotropicCorrection(classified, 'ice');
 
  var snowMask = classified.select('snow_mask');
  var mergedNB = TOPO_BANDS_ALL.map(function (band, idx) {
    var nbBand = 'narrowband_' + BAND_NUMS_ALL[idx];
    return nbBand === 'narrowband_b4' ? nbIce.select(nbBand) :
           nbIce.select(nbBand).where(snowMask, nbSnow.select(nbBand)).rename(nbBand);
  });
 
  var withBB = computeBroadbandAlbedo(classified.addBands(ee.Image.cat(mergedNB)), satellite);
  var glacierMask = createGlacierMask(glacierOutlines).reproject(withBB.select('broadband_albedo_' + satellite).projection());
 
  return filtered.addBands(withBB)
    .addBands(withBB.select('broadband_albedo_' + satellite).updateMask(glacierMask).rename('broadband_albedo_' + satellite + '_masked'))
    .copyProperties(image, ['system:time_start', 'system:time_end'])
    .set('qa_mode', MOD09GA_QA_CONFIG.mode).set('method', satellite);
}
// Process Collection for Multi-QA and Multi-Product
function processCollectionMulti(glacierData) {
  var initial = ee.ImageCollection([]);
  var results = {};
 
  // Process MOD09GA (Terra) with QA mode
  if (METHODS.indexOf('MOD09GA') !== -1) {
    var rawCollection = ee.ImageCollection(MODIS_COLLECTION)
      .filterDate(START_DATE, END_DATE)
      .filter(ee.Filter.calendarRange(START_MONTH, END_MONTH, 'month'))
      .filterBounds(glacierData.geometry);
   
    var processedMOD09GA = rawCollection.map(function(image) {
      return processRenMethod(image, glacierData.outlines, 'MOD09GA');
    });
    results['MOD09GA'] = processedMOD09GA;
  }
 
  // Process MYD09GA (Aqua) with QA mode
  if (METHODS.indexOf('MYD09GA') !== -1) {
    var rawCollectionAqua = ee.ImageCollection(AQUA_COLLECTION)
      .filterDate(START_DATE, END_DATE)
      .filter(ee.Filter.calendarRange(START_MONTH, END_MONTH, 'month'))
      .filterBounds(glacierData.geometry);
   
    var processedMYD09GA = rawCollectionAqua.map(function(image) {
      return processRenMethod(image, glacierData.outlines, 'MYD09GA');
    });
    results['MYD09GA'] = processedMYD09GA;
  }
 
  // Process MOD10A1 (Terra)
  if (METHODS.indexOf('mod10a1') !== -1) {
    var mod10Col = ee.ImageCollection(MODIS_COLLECTIONS.MOD10A1)
      .filterDate(START_DATE, END_DATE)
      .filter(ee.Filter.calendarRange(START_MONTH, END_MONTH, 'month'))
      .filterBounds(glacierData.geometry);
    results.mod10a1 = mod10Col.map(function(image) {
      return processMODIS10A1(image, glacierData.outlines, 'mod10a1');
    });
  }
 
  // Process MYD10A1 (Aqua)
  if (METHODS.indexOf('myd10a1') !== -1) {
    var myd10Col = ee.ImageCollection(MODIS_COLLECTIONS.MYD10A1)
      .filterDate(START_DATE, END_DATE)
      .filter(ee.Filter.calendarRange(START_MONTH, END_MONTH, 'month'))
      .filterBounds(glacierData.geometry);
    results.myd10a1 = myd10Col.map(function(image) {
      return processMODIS10A1(image, glacierData.outlines, 'myd10a1');
    });
  }
 
  // Process MCD43A3
  if (METHODS.indexOf('mcd43a3') !== -1) {
    var mcd43Col = ee.ImageCollection(MODIS_COLLECTIONS.MCD43A3)
      .filterDate(START_DATE, END_DATE)
      .filter(ee.Filter.calendarRange(START_MONTH, END_MONTH, 'month'))
      .filterBounds(glacierData.geometry);
    results.mcd43a3 = mcd43Col.map(function(image) {
      return processMCD43A3(image, glacierData.outlines);
    });
  }
 
  // Merge all collections
  var merged = initial;
  if (results['MOD09GA']) merged = merged.merge(results['MOD09GA']);
  if (results['MYD09GA']) merged = merged.merge(results['MYD09GA']);
  if (results.mod10a1) merged = merged.merge(results.mod10a1);
  if (results.myd10a1) merged = merged.merge(results.myd10a1);
  if (results.mcd43a3) merged = merged.merge(results.mcd43a3);
 
  print('Results object keys:', Object.keys(results));
  print('Merged collection size:', merged.size());
 
  return merged;
}
// Pixel Sampling (Modified to handle multiple methods including Terra and Aqua)
var SCALE_METERS = 500;
var MODIS_PROJ = ee.ImageCollection('MODIS/061/MOD09GA').first().select('sur_refl_b01').projection(); // Can also use MYD09GA
var DEM = dem.reproject({crs: MODIS_PROJ, scale: SCALE_METERS}).rename('elevation');
var SLOPE = slope.reproject({crs: MODIS_PROJ, scale: SCALE_METERS}).rename('slope');
var ASPECT = aspect.reproject({crs: MODIS_PROJ, scale: SCALE_METERS}).rename('aspect');

function addStablePixelCoords(image) {
  var coords = ee.Image.pixelCoordinates(MODIS_PROJ);
  var pixelRow = coords.select('y').toInt().rename('pixel_row');
  var pixelCol = coords.select('x').toInt().rename('pixel_col');
  var pixelId = pixelRow.multiply(1e6).add(pixelCol).toInt64().rename('pixel_id');
  var lonLat = ee.Image.pixelLonLat();
  var tileH = lonLat.select('longitude').multiply(100).round().toInt().rename('tile_h');
  var tileV = lonLat.select('latitude').multiply(100).round().toInt().rename('tile_v');
  return image.addBands([tileH, tileV, pixelRow, pixelCol, pixelId]);
}
function sampleDailyPixels(collection, region, glacierData) {
  // Create glacier fraction layer once for all methods
  var GLACIER_FRACTION = getGlacierFraction(glacierData.outlines).reproject({crs: MODIS_PROJ, scale: SCALE_METERS});
  // Split collection by method
  var MOD09GACollection = collection.filter(ee.Filter.eq('method', 'MOD09GA'));
  var MYD09GACollection = collection.filter(ee.Filter.eq('method', 'MYD09GA'));
  var mod10Collection = collection.filter(ee.Filter.eq('method', 'mod10a1'));
  var myd10Collection = collection.filter(ee.Filter.eq('method', 'myd10a1'));
  var mcd43Collection = collection.filter(ee.Filter.eq('method', 'mcd43a3'));
 
  // Debug: Check filtered collection sizes (only in debug mode)
  if (DEBUG_MODE) {
    print('In sampleDailyPixels - MOD09GA collection size:', MOD09GACollection.size());
    print('In sampleDailyPixels - MYD09GA collection size:', MYD09GACollection.size());
    print('In sampleDailyPixels - mod10a1 collection size:', mod10Collection.size());
    print('In sampleDailyPixels - myd10a1 collection size:', myd10Collection.size());
    print('In sampleDailyPixels - mcd43a3 collection size:', mcd43Collection.size());
  }
 
  // Process MOD09GA images
  var MOD09GASamples = MOD09GACollection.map(function(img) {
    var date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
    var qaMode = img.get('qa_mode');
    var method = 'MOD09GA';
   
    var base = addStablePixelCoords(img.select('broadband_albedo_MOD09GA_masked').rename('albedo'));
    base = base.addBands(DEM).addBands(SLOPE).addBands(ASPECT);
    base = base.addBands(GLACIER_FRACTION); // Add glacier fraction
    base = base.addBands(img.select('SolarZenith').multiply(0.01).rename('solar_zenith'));
    base = base.addBands(img.select('NDSI').rename('ndsi'));
   
    // Add reflectance bands
    REFL_BANDS.forEach(function(b) {
      base = base.addBands(img.select(b).multiply(0.0001).rename(b));
    });
   
    return base.sample({
      region: region,
      scale: SCALE_METERS,
      projection: MODIS_PROJ,
      geometries: true
    }).map(function(feat) {
      var coords = feat.geometry().coordinates();
      return ee.Feature(null, {
        'albedo': feat.get('albedo'),
        'elevation': feat.get('elevation'),
        'slope': feat.get('slope'),
        'aspect': feat.get('aspect'),
        'glacier_fraction': feat.get('glacier_fraction'),
        'solar_zenith': feat.get('solar_zenith'),
        'ndsi': feat.get('ndsi'),
        'sur_refl_b01': feat.get('sur_refl_b01'),
        'sur_refl_b02': feat.get('sur_refl_b02'),
        'sur_refl_b03': feat.get('sur_refl_b03'),
        'sur_refl_b04': feat.get('sur_refl_b04'),
        'sur_refl_b05': feat.get('sur_refl_b05'),
        'sur_refl_b06': feat.get('sur_refl_b06'),
        'sur_refl_b07': feat.get('sur_refl_b07'),
        'longitude': ee.List(coords).get(0),
        'latitude': ee.List(coords).get(1),
        'tile_h': feat.get('tile_h'),
        'tile_v': feat.get('tile_v'),
        'pixel_row': feat.get('pixel_row'),
        'pixel_col': feat.get('pixel_col'),
        'pixel_id': feat.get('pixel_id'),
        'date': date,
        'qa_mode': qaMode,
        'method': method
      });
    });
  }).flatten();
 
  // Debug: Check sample counts (only in debug mode)
  // if (DEBUG_MODE) print('MOD09GA samples size:', MOD09GASamples.size());
 
  // Process MYD09GA images (Aqua)
  var MYD09GASamples = MYD09GACollection.map(function(img) {
    var date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
    var qaMode = img.get('qa_mode');
    var method = 'MYD09GA';
   
    var base = addStablePixelCoords(img.select('broadband_albedo_MYD09GA_masked').rename('albedo'));
    base = base.addBands(DEM).addBands(SLOPE).addBands(ASPECT);
    base = base.addBands(GLACIER_FRACTION); // Add glacier fraction
    base = base.addBands(img.select('SolarZenith').multiply(0.01).rename('solar_zenith'));
    base = base.addBands(img.select('NDSI').rename('ndsi'));
   
    // Add reflectance bands
    REFL_BANDS.forEach(function(b) {
      base = base.addBands(img.select(b).multiply(0.0001).rename(b));
    });
   
    return base.sample({
      region: region,
      scale: SCALE_METERS,
      projection: MODIS_PROJ,
      geometries: true
    }).map(function(feat) {
      var coords = feat.geometry().coordinates();
      return ee.Feature(null, {
        'albedo': feat.get('albedo'),
        'elevation': feat.get('elevation'),
        'slope': feat.get('slope'),
        'aspect': feat.get('aspect'),
        'glacier_fraction': feat.get('glacier_fraction'),
        'solar_zenith': feat.get('solar_zenith'),
        'ndsi': feat.get('ndsi'),
        'sur_refl_b01': feat.get('sur_refl_b01'),
        'sur_refl_b02': feat.get('sur_refl_b02'),
        'sur_refl_b03': feat.get('sur_refl_b03'),
        'sur_refl_b04': feat.get('sur_refl_b04'),
        'sur_refl_b05': feat.get('sur_refl_b05'),
        'sur_refl_b06': feat.get('sur_refl_b06'),
        'sur_refl_b07': feat.get('sur_refl_b07'),
        'longitude': ee.List(coords).get(0),
        'latitude': ee.List(coords).get(1),
        'tile_h': feat.get('tile_h'),
        'tile_v': feat.get('tile_v'),
        'pixel_row': feat.get('pixel_row'),
        'pixel_col': feat.get('pixel_col'),
        'pixel_id': feat.get('pixel_id'),
        'date': date,
        'qa_mode': qaMode,
        'method': method
      });
    });
  }).flatten();
 
  // Debug: Check sample counts (only in debug mode)
  // if (DEBUG_MODE) print('MYD09GA samples size:', MYD09GASamples.size());
 
  // Process MOD10A1 images
  var mod10Samples = mod10Collection.map(function(img) {
    var date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
    var qaMode = img.get('qa_mode');
    var method = 'mod10a1';
   
    var base = addStablePixelCoords(img.select('broadband_albedo_mod10a1_masked').rename('albedo'));
    base = base.addBands(DEM).addBands(SLOPE).addBands(ASPECT);
    base = base.addBands(GLACIER_FRACTION); // Add glacier fraction
    base = base.addBands(img.select('NDSI').rename('ndsi'));
   
    return base.sample({
      region: region,
      scale: SCALE_METERS,
      projection: MODIS_PROJ,
      geometries: true
    }).map(function(feat) {
      var coords = feat.geometry().coordinates();
      return ee.Feature(null, {
        'albedo': feat.get('albedo'),
        'elevation': feat.get('elevation'),
        'slope': feat.get('slope'),
        'aspect': feat.get('aspect'),
        'glacier_fraction': feat.get('glacier_fraction'),
        'solar_zenith': -999,
        'ndsi': feat.get('ndsi'),
        'sur_refl_b01': 'n/a',
        'sur_refl_b02': 'n/a',
        'sur_refl_b03': 'n/a',
        'sur_refl_b04': 'n/a',
        'sur_refl_b05': 'n/a',
        'sur_refl_b06': 'n/a',
        'sur_refl_b07': 'n/a',
        'longitude': ee.List(coords).get(0),
        'latitude': ee.List(coords).get(1),
        'tile_h': feat.get('tile_h'),
        'tile_v': feat.get('tile_v'),
        'pixel_row': feat.get('pixel_row'),
        'pixel_col': feat.get('pixel_col'),
        'pixel_id': feat.get('pixel_id'),
        'date': date,
        'qa_mode': qaMode,
        'method': method
      });
    });
  }).flatten();
 
  // Debug: Check sample counts (only in debug mode)
  // if (DEBUG_MODE) print('mod10a1 samples size:', mod10Samples.size());
 
  // Process MYD10A1 images (Aqua)
  var myd10Samples = myd10Collection.map(function(img) {
    var date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
    var qaMode = img.get('qa_mode');
    var method = 'myd10a1';
   
    var base = addStablePixelCoords(img.select('broadband_albedo_myd10a1_masked').rename('albedo'));
    base = base.addBands(DEM).addBands(SLOPE).addBands(ASPECT);
    base = base.addBands(GLACIER_FRACTION); // Add glacier fraction
    base = base.addBands(img.select('NDSI').rename('ndsi'));
   
    return base.sample({
      region: region,
      scale: SCALE_METERS,
      projection: MODIS_PROJ,
      geometries: true
    }).map(function(feat) {
      var coords = feat.geometry().coordinates();
      return ee.Feature(null, {
        'albedo': feat.get('albedo'),
        'elevation': feat.get('elevation'),
        'slope': feat.get('slope'),
        'aspect': feat.get('aspect'),
        'glacier_fraction': feat.get('glacier_fraction'),
        'solar_zenith': -999,
        'ndsi': feat.get('ndsi'),
        'sur_refl_b01': 'n/a',
        'sur_refl_b02': 'n/a',
        'sur_refl_b03': 'n/a',
        'sur_refl_b04': 'n/a',
        'sur_refl_b05': 'n/a',
        'sur_refl_b06': 'n/a',
        'sur_refl_b07': 'n/a',
        'longitude': ee.List(coords).get(0),
        'latitude': ee.List(coords).get(1),
        'tile_h': feat.get('tile_h'),
        'tile_v': feat.get('tile_v'),
        'pixel_row': feat.get('pixel_row'),
        'pixel_col': feat.get('pixel_col'),
        'pixel_id': feat.get('pixel_id'),
        'date': date,
        'qa_mode': qaMode,
        'method': method
      });
    });
  }).flatten();
 
  // Debug: Check sample counts (only in debug mode)
  // if (DEBUG_MODE) print('myd10a1 samples size:', myd10Samples.size());
 
  // Process MCD43A3 images
  var mcd43Samples = mcd43Collection.map(function(img) {
    var date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
    var qaMode = img.get('qa_mode');
    var method = 'mcd43a3';
   
    var base = addStablePixelCoords(img.select('broadband_albedo_mcd43a3_masked').rename('albedo'));
    base = base.addBands(DEM).addBands(SLOPE).addBands(ASPECT);
    base = base.addBands(GLACIER_FRACTION); // Add glacier fraction
   
    return base.sample({
      region: region,
      scale: SCALE_METERS,
      projection: MODIS_PROJ,
      geometries: true
    }).map(function(feat) {
      var coords = feat.geometry().coordinates();
      return ee.Feature(null, {
        'albedo': feat.get('albedo'),
        'elevation': feat.get('elevation'),
        'slope': feat.get('slope'),
        'aspect': feat.get('aspect'),
        'glacier_fraction': feat.get('glacier_fraction'),
        'solar_zenith': -999,
        'ndsi': -999,
        'sur_refl_b01': 'n/a',
        'sur_refl_b02': 'n/a',
        'sur_refl_b03': 'n/a',
        'sur_refl_b04': 'n/a',
        'sur_refl_b05': 'n/a',
        'sur_refl_b06': 'n/a',
        'sur_refl_b07': 'n/a',
        'longitude': ee.List(coords).get(0),
        'latitude': ee.List(coords).get(1),
        'tile_h': feat.get('tile_h'),
        'tile_v': feat.get('tile_v'),
        'pixel_row': feat.get('pixel_row'),
        'pixel_col': feat.get('pixel_col'),
        'pixel_id': feat.get('pixel_id'),
        'date': date,
        'qa_mode': qaMode,
        'method': method
      });
    });
  }).flatten();
 
  // Debug: Check sample counts (only in debug mode)
  // if (DEBUG_MODE) print('mcd43a3 samples size:', mcd43Samples.size());
 
  // Merge all samples
  return MOD09GASamples.merge(MYD09GASamples).merge(mod10Samples).merge(myd10Samples).merge(mcd43Samples);
}

// Create pivoted pixel data (wide format)
function createPivotedPixelData(sampledPixels) {
  // Group by pixel_id and date
  var uniqueKeys = sampledPixels.distinct(['pixel_id', 'date']);
  
  // Function to get value for specific method
  var pivoted = uniqueKeys.map(function(key) {
    var pixelId = key.get('pixel_id');
    var date = key.get('date');
    
    // Filter all samples for this pixel-date
    var pixelDateSamples = sampledPixels
      .filter(ee.Filter.eq('pixel_id', pixelId))
      .filter(ee.Filter.eq('date', date));
    
    // Get first sample for common attributes
    var first = ee.Feature(pixelDateSamples.first());
    
    // Initialize properties with common attributes
    var properties = {
      'pixel_id': pixelId,
      'date': date,
      'longitude': first.get('longitude'),
      'latitude': first.get('latitude'),
      'elevation': first.get('elevation'),
      'slope': first.get('slope'),
      'aspect': first.get('aspect'),
      'glacier_fraction': first.get('glacier_fraction'),
      'tile_h': first.get('tile_h'),
      'tile_v': first.get('tile_v'),
      'pixel_row': first.get('pixel_row'),
      'pixel_col': first.get('pixel_col')
    };
    
    // Process each method
    var methods = ['MOD09GA', 'MYD09GA', 'mod10a1', 'myd10a1', 'mcd43a3'];
    
    methods.forEach(function(method) {
      var methodSample = pixelDateSamples.filter(ee.Filter.eq('method', method));
      var hasData = methodSample.size().gt(0);
      
      // Get values or set to null
      var sample = ee.Feature(ee.Algorithms.If(hasData, methodSample.first(), 
                                                ee.Feature(null, {'albedo': -999, 'ndsi': -999, 'solar_zenith': -999})));
      
      // Albedo (all methods)
      properties['albedo_' + method] = ee.Algorithms.If(hasData, sample.get('albedo'), -999);
      
      // NDSI (not for MCD43A3)
      if (method !== 'mcd43a3') {
        properties['ndsi_' + method] = ee.Algorithms.If(hasData, sample.get('ndsi'), -999);
      }
      
      // Solar zenith and reflectances (only MOD09GA and MYD09GA)
      if (method === 'MOD09GA' || method === 'MYD09GA') {
        properties['solar_zenith_' + method] = ee.Algorithms.If(hasData, sample.get('solar_zenith'), -999);
        
        // Add reflectance bands
        var bands = ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04', 
                     'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07'];
        bands.forEach(function(band) {
          properties[band + '_' + method] = ee.Algorithms.If(hasData, sample.get(band), -999);
        });
      }
    });
    
    return ee.Feature(null, properties);
  });
  
  return pivoted;
}

// Create daily summary statistics
function createDailySummary(sampledPixels) {
  // Get unique dates
  var uniqueDates = sampledPixels.distinct(['date']);
  
  var dailySummary = uniqueDates.map(function(dateFeature) {
    var date = dateFeature.get('date');
    
    // Filter all samples for this date
    var dateSamples = sampledPixels.filter(ee.Filter.eq('date', date));
    
    // Initialize summary properties
    var summaryProps = {'date': date};
    
    // Calculate statistics for each method
    var methods = ['MOD09GA', 'MYD09GA', 'mod10a1', 'myd10a1', 'mcd43a3'];
    
    methods.forEach(function(method) {
      var methodSamples = dateSamples.filter(ee.Filter.eq('method', method));
      var hasData = methodSamples.size().gt(0);
      
      // Calculate mean, std, and count
      var stats = ee.Algorithms.If(
        hasData,
        methodSamples.reduceColumns({
          reducer: ee.Reducer.mean().combine({
            reducer2: ee.Reducer.stdDev(),
            sharedInputs: true
          }).combine({
            reducer2: ee.Reducer.count(),
            sharedInputs: true
          }),
          selectors: ['albedo']
        }),
        ee.Dictionary({'mean': -999, 'stdDev': -999, 'count': 0})
      );
      
      summaryProps['albedo_mean_' + method] = ee.Dictionary(stats).get('mean', -999);
      summaryProps['albedo_std_' + method] = ee.Dictionary(stats).get('stdDev', -999);
      summaryProps['pixel_count_' + method] = ee.Dictionary(stats).get('count', 0);
    });
    
    // Calculate mean glacier fraction for the day
    var glacierStats = dateSamples.reduceColumns({
      reducer: ee.Reducer.mean(),
      selectors: ['glacier_fraction']
    });
    summaryProps['glacier_fraction_mean'] = ee.Dictionary(glacierStats).get('mean', -999);
    
    return ee.Feature(null, summaryProps);
  });
  
  return dailySummary;
}
// Export Functions
// Export 1: Long format (original structure)
function exportPixelCSVLong(sampledPixels, glacierName) {
  var desc = glacierName + '_MODIS_PixelLevel_Long_' + START_DATE + '_to_' + END_DATE;
  Export.table.toDrive({
    collection: sampledPixels,
    description: desc,
    folder: EXPORT_CONFIG.folder,
    fileFormat: 'CSV',
    selectors: [
      'pixel_id', 'date', 'qa_mode', 'method', 'albedo', 'glacier_fraction', 'ndsi', 'solar_zenith',
      'elevation', 'slope', 'aspect', 'longitude', 'latitude',
      'tile_h', 'tile_v', 'pixel_row', 'pixel_col',
      'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04',
      'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07'
    ]
  });
  print('Export 1 initiated: ' + desc);
}

// Export 2: Pivoted pixel-level format
function exportPixelCSVPivoted(pivotedData, glacierName) {
  var desc = glacierName + '_MODIS_PixelLevel_Pivoted_' + START_DATE + '_to_' + END_DATE;
  
  // Build selectors dynamically based on available columns
  var baseSelectors = ['pixel_id', 'date', 'longitude', 'latitude', 'elevation', 'slope', 'aspect', 
                       'glacier_fraction', 'tile_h', 'tile_v', 'pixel_row', 'pixel_col'];
  
  // Add albedo columns for all methods
  var albedoSelectors = ['albedo_MOD09GA', 'albedo_MYD09GA', 'albedo_mod10a1', 'albedo_myd10a1', 'albedo_mcd43a3'];
  
  // Add NDSI columns (not for MCD43A3)
  var ndsiSelectors = ['ndsi_MOD09GA', 'ndsi_MYD09GA', 'ndsi_mod10a1', 'ndsi_myd10a1'];
  
  // Add solar zenith (only MOD09GA and MYD09GA)
  var solarSelectors = ['solar_zenith_MOD09GA', 'solar_zenith_MYD09GA'];
  
  // Add reflectance bands (only MOD09GA and MYD09GA)
  var reflectanceSelectors = [];
  var bands = ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04', 
               'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07'];
  bands.forEach(function(band) {
    reflectanceSelectors.push(band + '_MOD09GA');
    reflectanceSelectors.push(band + '_MYD09GA');
  });
  
  var allSelectors = baseSelectors.concat(albedoSelectors).concat(ndsiSelectors)
                                   .concat(solarSelectors).concat(reflectanceSelectors);
  
  Export.table.toDrive({
    collection: pivotedData,
    description: desc,
    folder: EXPORT_CONFIG.folder,
    fileFormat: 'CSV',
    selectors: allSelectors
  });
  print('Export 2 initiated: ' + desc);
}

// Export 3: Daily summary format
function exportDailySummaryCSV(dailySummary, glacierName) {
  var desc = glacierName + '_MODIS_DailySummary_' + START_DATE + '_to_' + END_DATE;
  
  // Build selectors for summary statistics
  var selectors = ['date'];
  
  // Add statistics for each method
  var methods = ['MOD09GA', 'MYD09GA', 'mod10a1', 'myd10a1', 'mcd43a3'];
  methods.forEach(function(method) {
    selectors.push('albedo_mean_' + method);
    selectors.push('albedo_std_' + method);
    selectors.push('pixel_count_' + method);
  });
  
  selectors.push('glacier_fraction_mean');
  
  Export.table.toDrive({
    collection: dailySummary,
    description: desc,
    folder: EXPORT_CONFIG.folder,
    fileFormat: 'CSV',
    selectors: selectors
  });
  print('Export 3 initiated: ' + desc);
}
// Main Processing Function
function processAndExportGlacier(assetPath) {
  var glacierName = extractGlacierName(assetPath);
  var glacierData = initializeGlacierData(assetPath);
  
  print('===================================');
  print('Processing glacier: ' + glacierName);
  print('Asset: ' + assetPath);
  print('===================================');
  
  // Process data
  var processedCollection = processCollectionMulti(glacierData);
  
  // Only show detailed counts in debug mode
  if (DEBUG_MODE) {
    // Check collection sizes
    print('Checking collection sizes for ' + glacierName + '...');
    var MOD09GAImages = processedCollection.filter(ee.Filter.eq('method', 'MOD09GA'));
    var MYD09GAImages = processedCollection.filter(ee.Filter.eq('method', 'MYD09GA'));
    var mod10Images = processedCollection.filter(ee.Filter.eq('method', 'mod10a1'));
    var myd10Images = processedCollection.filter(ee.Filter.eq('method', 'myd10a1'));
    var mcd43Images = processedCollection.filter(ee.Filter.eq('method', 'mcd43a3'));
    
    print('MOD09GA images:', MOD09GAImages.size());
    print('MYD09GA images:', MYD09GAImages.size());
    print('MOD10A1 images:', mod10Images.size());
    print('MYD10A1 images:', myd10Images.size());
    print('MCD43A3 images:', mcd43Images.size());
  } else {
    print('Processing all methods...');
  }
  
  // Sample pixels
  var sampledPixels = sampleDailyPixels(processedCollection, glacierData.geometry, glacierData);
  
  // Skip detailed pixel counts to avoid memory issues
  if (DEBUG_MODE) {
    print('Sampled pixel counts by method:');
    print('MOD09GA:', sampledPixels.filter(ee.Filter.eq('method', 'MOD09GA')).size());
    print('MYD09GA:', sampledPixels.filter(ee.Filter.eq('method', 'MYD09GA')).size());
    print('mod10a1:', sampledPixels.filter(ee.Filter.eq('method', 'mod10a1')).size());
    print('myd10a1:', sampledPixels.filter(ee.Filter.eq('method', 'myd10a1')).size());
    print('mcd43a3:', sampledPixels.filter(ee.Filter.eq('method', 'mcd43a3')).size());
  }
  
  // Create pivoted and summary formats
  if (EXPORT_CONFIG.formats.pivoted) {
    print('Creating pivoted pixel data...');
    var pivotedData = createPivotedPixelData(sampledPixels);
  }
  
  if (EXPORT_CONFIG.formats.summary) {
    print('Creating daily summary...');
    var dailySummary = createDailySummary(sampledPixels);
  }
  
  // Export formats based on configuration
  if (EXPORT_CONFIG.formats.long) {
    exportPixelCSVLong(sampledPixels, glacierName);
  }
  if (EXPORT_CONFIG.formats.pivoted) {
    exportPixelCSVPivoted(pivotedData, glacierName);
  }
  if (EXPORT_CONFIG.formats.summary) {
    exportDailySummaryCSV(dailySummary, glacierName);
  }
  
  // Glacier statistics
  var glacierMask = createGlacierMask(glacierData.outlines);
  var glacierMaskProjected = glacierMask.reproject({crs: MODIS_PROJ, scale: SCALE_METERS});
  
  var glacierAreaKm2 = glacierMaskProjected.multiply(ee.Image.pixelArea())
    .divide(1e6)
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: glacierData.geometry,
      scale: SCALE_METERS,
      maxPixels: 1e9
    });
  
  if (DEBUG_MODE) {
    print('Glacier area (km²):', glacierAreaKm2);
  }
  
  // Visualization (only for first glacier in single mode)
  if (SINGLE_GLACIER_MODE) {
    Map.centerObject(glacierData.geometry, 11);
    Map.addLayer(glacierData.image, {min: 0, max: 1, palette: ['white', 'blue']}, 
                 glacierName + ' Original Asset', false);
    Map.addLayer(glacierMaskProjected, {min: 0, max: 1, palette: ['white', 'red']}, 
                 glacierName + ' MODIS Mask (>50%)', true);
    
    // Add glacier fraction visualization
    var GLACIER_FRACTION = getGlacierFraction(glacierData.outlines).reproject({crs: MODIS_PROJ, scale: SCALE_METERS});
    Map.addLayer(GLACIER_FRACTION, {min: 0, max: 1, palette: ['white', 'yellow', 'orange', 'red']}, 
                 glacierName + ' Glacier Fraction (0-1)', true);
    
    // Skip sample albedo visualization to avoid memory issues unless in debug mode
    if (DEBUG_MODE) {
      var MOD09GAImages = processedCollection.filter(ee.Filter.eq('method', 'MOD09GA'));
      MOD09GAImages.size().evaluate(function(size) {
        if (size > 0) {
          var sample = MOD09GAImages.first();
          Map.addLayer(sample.select('broadband_albedo_MOD09GA_masked'), 
                       {min: 0, max: 1, palette: ['black', 'white']},
                       glacierName + ' Sample MOD09GA Albedo', false);
        }
      });
    }
  }
  
  print('Processing complete for ' + glacierName);
  print('');
}

// ==============================================================================
// MAIN EXECUTION
// ==============================================================================

print('MODIS Multi-Product Albedo Analysis');
print('Date range: ' + START_DATE + ' to ' + END_DATE);
print('Months: ' + START_MONTH + ' to ' + END_MONTH);
print('');

if (SINGLE_GLACIER_MODE) {
  // Process single glacier
  print('Mode: SINGLE GLACIER');
  processAndExportGlacier(GLACIER_ASSET);
} else {
  // Process multiple glaciers
  print('Mode: MULTIPLE GLACIERS');
  print('Number of glaciers: ' + GLACIER_ASSETS.length);
  print('');
  
  GLACIER_ASSETS.forEach(function(assetPath) {
    processAndExportGlacier(assetPath);
  });
}

print('===================================');
print('EXPORT SUMMARY');
print('===================================');
print('Three export formats per glacier:');
print('1. Long format: Original pixel-level structure');
print('2. Pivoted format: Wide format with method columns');
print('3. Daily summary: Aggregated statistics');
print('');
print('Export folder: ' + EXPORT_CONFIG.folder);
print('');
print('Column descriptions:');
print('- glacier_fraction: 0.50-1.00 (50-100% glacier coverage)');
print('- Albedo values: 0-1 scale');
print('- Missing values: -999 or NaN');
print('- Reflectances: Only for MOD09GA/MYD09GA');
print('');
print('Methods included:');
print('- MOD09GA: Terra morning (~10:30 AM) with Ren processing');
print('- MYD09GA: Aqua afternoon (~1:30 PM) with Ren processing');
print('- mod10a1: Terra morning snow albedo product');
print('- myd10a1: Aqua afternoon snow albedo product');
print('- mcd43a3: 16-day BRDF-corrected albedo');