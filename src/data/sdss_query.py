from astroquery.sdss import SDSS

def get_sdss_data(limit=20000):
    """Obtain data from SDSS with a limit on the number of samples."""
    query = f"""
    SELECT TOP {limit}
      objid, ra, dec, type,
      psfMag_u, psfMag_g, psfMag_r, psfMag_i, psfMag_z,
      modelMag_u, modelMag_g, modelMag_r, modelMag_i, modelMag_z, petroR50_r, petroR90_r
    FROM PhotoObj
    WHERE type IN (3, 6) AND clean = 1
    """
    data = SDSS.query_sql(query)
    df = data.to_pandas()

    return df
