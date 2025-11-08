import osmnx as ox
import folium

place_name = "Granada, Spain"
G = ox.graph_from_place(place_name, network_type="drive")

nodes, edges = ox.graph_to_gdfs(G)

center = [nodes.geometry.y.mean(), nodes.geometry.x.mean()]
m = folium.Map(location=center, zoom_start=13)

for _, row in edges.iterrows():
    folium.PolyLine(
        locations=[(y, x) for x, y in row.geometry.coords],
        color="blue", weight=2, opacity=0.7
    ).add_to(m)

m.save("map.html")

