from astroquery.sdss import SDSS

def get_sdss_data(offset=0, limit=20000):
    """Obtain data from SDSS with a limit on the number of samples."""
    query = f"""
    SELECT *
    FROM (
        SELECT
            objid, ra, dec, type,
            psfMag_u, psfMag_g, psfMag_r, psfMag_i, psfMag_z,
            modelMag_u, modelMag_g, modelMag_r, modelMag_i, modelMag_z,
            petroR50_r, petroR90_r,
            ROW_NUMBER() OVER (ORDER BY objid) AS rn
        FROM PhotoObj
        WHERE type IN (3, 6) AND clean = 1
    ) AS sub
    WHERE rn > {offset} AND rn <= {offset + limit}
    """
    data = SDSS.query_sql(query)
    df = data.to_pandas()

    return df
