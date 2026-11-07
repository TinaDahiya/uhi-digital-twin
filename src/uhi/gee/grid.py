"""Spec section 5: regular analysis grid in the city's UTM zone."""
import math


def build_grid(ee, aoi, epsg: int, cell_m: int = 1000):
    from pyproj import Transformer
    ring = aoi.bounds().coordinates().getInfo()[0]
    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    xs, ys = zip(*[to_utm.transform(x, y) for x, y in zip(lons, lats)])
    ix0, ix1 = math.floor(min(xs) / cell_m), math.ceil(max(xs) / cell_m)
    iy0, iy1 = math.floor(min(ys) / cell_m), math.ceil(max(ys) / cell_m)
    feats = []
    for ix in range(ix0, ix1):
        for iy in range(iy0, iy1):
            x0, y0 = ix * cell_m, iy * cell_m
            cx, cy = to_ll.transform(x0 + cell_m / 2, y0 + cell_m / 2)
            geom = ee.Geometry.Rectangle([x0, y0, x0 + cell_m, y0 + cell_m], proj=f"EPSG:{epsg}", geodesic=False)
            feats.append(ee.Feature(geom, {"cell_id": f"{ix}_{iy}", "lon": cx, "lat": cy}))
    return ee.FeatureCollection(feats).filterBounds(aoi), len(feats)
