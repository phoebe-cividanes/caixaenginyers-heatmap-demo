import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from geopy.distance import distance as geodist
from geopy.geocoders import Nominatim
import folium
import random

# ----------------------------
# 1. Geocode municipalities
# ----------------------------
keys = ["Granada, Spain", "Jaen, Spain", "Cordoba, Spain", "El Bayo, Spain", "Àgreda, Spain", "Fuenmayor, Spain"]
geolocator = Nominatim(user_agent="geoapi")

locations = []
for key in keys:
    loc = geolocator.geocode(key)
    if loc:
        locations.append((key, loc.latitude, loc.longitude))
    else:
        print(f"Warning: {key} not found.")

if not locations:
    raise ValueError("No valid municipalities found.")

gdf = gpd.GeoDataFrame(locations, columns=["name", "lat", "lon"])
gdf["geometry"] = gdf.apply(lambda r: Point(r.lon, r.lat), axis=1)

# ----------------------------
# 2. Build neighbor graph (≤100 km)
# ----------------------------
G = nx.Graph()
for i, a in gdf.iterrows():
    for j, b in gdf.iterrows():
        if i >= j:
            continue
        d = geodist((a.lat, a.lon), (b.lat, b.lon)).km
        if d <= 100:
            G.add_edge(a.name, b.name, weight=d)

# ----------------------------
# 3. Assign a random color to each connected component
# ----------------------------
components = list(nx.connected_components(G))
colors = {}
for comp in components:
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    for name in comp:
        colors[name] = color

# ----------------------------
# 4. Plot on a Folium map (no edges)
# ----------------------------
center = (gdf.lat.mean(), gdf.lon.mean())
m = folium.Map(location=center, zoom_start=8)

for _, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=10,
        color=colors.get(row.name, "#000000"),
        fill=True,
        fill_color=colors.get(row.name, "#000000"),
        fill_opacity=0.7,
        popup=row.name
    ).add_to(m)

# ----------------------------
# 5. Save map
# ----------------------------
m.save("municipalities_components.html")
print("Map saved to municipalities_components.html — open it in your browser.")

