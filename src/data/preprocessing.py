

def process_sdss_data(df):
    # Creating colour indices
    df['psfMag_(u-g)'] = df['psfMag_u'] - df['psfMag_g']
    df['psfMag_(g-r)'] = df['psfMag_g'] - df['psfMag_r']

    df['psfMag_(r-i)'] = df['psfMag_r'] - df['psfMag_i']
    df['psfMag_(i-z)'] = df['psfMag_i'] - df['psfMag_z']

    df['modelMag_(u-g)'] = df['modelMag_u'] - df['modelMag_g']
    df['modelMag_(g-r)'] = df['modelMag_g'] - df['modelMag_r']

    df['modelMag_(r-i)'] = df['modelMag_r'] - df['modelMag_i']
    df['modelMag_(i-z)'] = df['modelMag_i'] - df['modelMag_z']

    # Creating psf-model in the r band
    df['psf-model'] = df['psfMag_r'] - df['modelMag_r']

    # Creating concentration index
    df['concentration_r'] = df['petroR90_r']/df['petroR50_r']

    type_map = {3: 'Galaxy', 6: 'Star'}
    df['class'] = df['type'].map(type_map)
    df = df.drop(columns=['objid', 'ra', 'dec', 'type', 'rn'], errors='ignore')

    return df