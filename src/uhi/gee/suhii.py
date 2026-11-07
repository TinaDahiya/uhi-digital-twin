"""Stage 2 / spec section 4: pixel-level SUHII against a LOCAL rural reference."""


def compute_suhii(ee, comp, suhii_cfg: dict, crs: str):
    water = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
             .unmask(0).gt(suhii_cfg["water_occurrence_min"]))          # unmask(0) MUST stay
    rural = (comp.select("NDBI").lte(suhii_cfg["rural_ndbi_max"])
             .And(comp.select("NDVI").gt(suhii_cfg["rural_ndvi_min"]))
             .And(water.Not()))
    lst_rural = comp.select("LST").updateMask(rural)
    ref_scale = suhii_cfg.get("reference_scale_m", 30)
    if ref_scale and ref_scale > 30:  # optional compute fallback (changes resolution of the reference only)
        lst_rural = lst_rural.reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(crs=crs, scale=ref_scale)
    ref = lst_rural.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.circle(radius=suhii_cfg["radius_m"], units="meters"),
        skipMasked=False,                                                # MUST stay
    ).rename("LST_ref")
    suhii = comp.select("LST").subtract(ref).rename("SUHII")
    return ee.Image.cat([suhii, ref])
