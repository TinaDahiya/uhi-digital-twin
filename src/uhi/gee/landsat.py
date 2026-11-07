"""Spec 3.1-3.3: Landsat 8/9 C2 L2 -> LST, NDVI, NDBI, NDWI, BSI, Albedo (per scene, then median)."""

QA_BITS = (0, 1, 2, 3, 4)  # fill, dilated cloud, cirrus, cloud, shadow
SR_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
# Liang-type broadband albedo, mapped OLI B2/B4/B5/B6/B7 <- ETM+ B1/B3/B4/B5/B7.
# OPEN ITEM (spec 3.3): confirm the OLI coefficient set against a citation before publication.
LIANG = {"blue": 0.356, "red": 0.130, "nir": 0.373, "swir1": 0.085, "swir2": 0.072, "intercept": -0.0018}


def mask_qa(ee, img):
    qa = img.select("QA_PIXEL")
    m = ee.Image(1)
    for b in QA_BITS:
        m = m.And(qa.bitwiseAnd(1 << b).eq(0))
    return img.updateMask(m)


def prep_scene(ee, img):
    img = mask_qa(ee, img)
    lst = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("LST")
    sr = img.select(SR_BANDS).multiply(0.0000275).add(-0.2)
    blue, green, red, nir, swir1, swir2 = (sr.select(b) for b in SR_BANDS)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("NDBI")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
    num = swir1.add(red).subtract(nir.add(blue))
    den = swir1.add(red).add(nir.add(blue))
    bsi = num.divide(den).rename("BSI")
    albedo = (blue.multiply(LIANG["blue"]).add(red.multiply(LIANG["red"]))
              .add(nir.multiply(LIANG["nir"])).add(swir1.multiply(LIANG["swir1"]))
              .add(swir2.multiply(LIANG["swir2"])).add(LIANG["intercept"])
              .clamp(0, 1).rename("Albedo"))
    return ee.Image.cat([lst, ndvi, ndbi, ndwi, bsi, albedo]).copyProperties(img, ["system:time_start"])


def landsat_collection(ee, aoi, start, end_excl, max_cloud=40):
    """Merged, cloud-filtered L8+L9 collection (raw scenes)."""
    cols = [ee.ImageCollection(c) for c in ("LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2")]
    col = cols[0].merge(cols[1])
    return (col.filterBounds(aoi).filterDate(start.isoformat(), end_excl.isoformat())
            .filter(ee.Filter.lt("CLOUD_COVER", max_cloud)))


def landsat_composite(ee, aoi, start, end_excl, max_cloud=40):
    col = landsat_collection(ee, aoi, start, end_excl, max_cloud)
    return col.map(lambda i: prep_scene(ee, i)).median(), col.size()
