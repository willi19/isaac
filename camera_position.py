import open3d as o3d
import numpy as np


def create_sphere(center, radius):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.7, 0.1, 0.1])  # 구의 색상 (빨간색)
    return sphere


convex_hull = o3d.io.read_triangle_mesh("rsc/convexhull/convex_hull_mesh.obj")
# Z-up → Y-up 변환 (X, Y, Z → X, Z, -Y)
R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
vertices = np.asarray(convex_hull.vertices)
transformed_vertices = vertices @ R.T

# 변환된 좌표를 다시 메쉬에 적용
convex_hull.vertices = o3d.utility.Vector3dVector(transformed_vertices)

# aabb = convex_hull.get_axis_aligned_bounding_box()
# min_bound = aabb.get_min_bound()  # 최소 좌표 (x_min, y_min, z_min)
# max_bound = aabb.get_max_bound()  # 최대 좌표 (x_max, y_max, z_max)
# extent = aabb.get_extent()  # 직육면체의 크기 (x_size, y_size, z_size)
# center = aabb.get_center()  # 중심 좌표
# aabb.color = (1, 0, 0)

# print("Bounding Box 정보:")
# print(f"- 최소 좌표: {min_bound}")
# print(f"- 최대 좌표: {max_bound}")
# print(f"- 크기 (Extent): {extent}")
# print(f"- 중심 (Center): {center}")
# 직육면체의 길이 설정 (X, Y, Z)
box_length = [0.6, 1.0, 1.4]

# 직육면체 생성 (Axis-Aligned Bounding Box로 생성)
min_bound = [-0.2, 0, -0.8]
max_bound = [1, 1.0, 0.8]
aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
aabb.color = (0, 0, 1)  # 파란색으로 설정

# 축 생성
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=[0, 0, 0]
)

axis_length = 0.6
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=axis_length, origin=[0, 0, 0]
)

cam_sphere = []
x_range = np.linspace(min_bound[0], max_bound[0], 7)
y_range = np.linspace(min_bound[1] + 0.2, max_bound[1], 5)
z_range = np.linspace(min_bound[2], max_bound[2], 9)
print(x_range)
print(y_range)
print(z_range)
for x in x_range:
    for y in y_range:
        for z in [min_bound[2], max_bound[2]]:
            cam_sphere.append(create_sphere([x, y, z], 0.02))

for x in x_range:
    for z in z_range:
        for y in [max_bound[1]]:
            cam_sphere.append(create_sphere([x, y, z], 0.02))

for y in y_range:
    for z in z_range:
        for x in [min_bound[0], max_bound[0]]:
            cam_sphere.append(create_sphere([x, y, z], 0.02))

o3d.visualization.draw_geometries([convex_hull, aabb, mesh_frame] + cam_sphere)
