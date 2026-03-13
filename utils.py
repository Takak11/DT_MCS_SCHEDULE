import math


def haversine_distance(coord1, coord2):
    """
    计算两经纬度坐标点之间的距离 (单位: km)
    coord1, coord2 格式为 (纬度 latitude, 经度 longitude)
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0 # 地球平均半径 (公里)

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance
