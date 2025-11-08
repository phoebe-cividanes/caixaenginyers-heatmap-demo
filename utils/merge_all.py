# utils/merge_all.py
import pandas as pd

def main(out_csv="data/merged_es.csv"):
    muni = pd.read_csv("data/muni_centroids.csv")  # muni_id, muni_nombre, lat, lon, prov_id, ca_id, area_km2
    ine  = pd.read_csv("data/demographics.csv")    # poblacion, renta_media, crecimiento_pob_5a
    macro= pd.read_csv("data/macro_by_muni.csv")   # pib_pc, digital
    rent = pd.read_csv("data/alquiler.csv")        # alquiler_m2
    osm  = pd.read_csv("data/osm_by_muni.csv")     # dens_bancos_km2, dist_min_banco_min
    incl = pd.read_csv("data/inclusion_by_muni.csv")# inclusion

    df = muni.merge(ine, on="muni_id", how="left")\
             .merge(macro, on="muni_id", how="left")\
             .merge(rent, on="muni_id", how="left")\
             .merge(osm, on="muni_id", how="left")\
             .merge(incl, on="muni_id", how="left")

    # valores por defecto si faltan
    for c in ["poblacion","renta_media","crecimiento_pob_5a","pib_pc","digital","alquiler_m2",
              "dens_bancos_km2","dist_min_banco_min","inclusion"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    df.to_csv(out_csv, index=False)
    print("OK ->", out_csv)

if __name__ == "__main__":
    main()
