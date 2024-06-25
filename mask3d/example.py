import numpy as np
import torch
from mask3d import get_model, load_mesh, prepare_data, map_output_to_pointcloud, save_colorized_mesh 

def run_for_scene(model, ply_file):

    # load input data
    # let's flatten in ...
    pointcloud_file = f"data/ply_files/{ply_file}"
    mesh = load_mesh(pointcloud_file)

    # prepare data
    data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)

    # run model
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)

    # map output to point cloud
    labels = map_output_to_pointcloud(mesh, outputs, inverse_map)

    fn = f"data/labeled/{ply_file[:-4]}_labels.npy"
    with open(fn, 'wb') as f:
        np.save(f, labels)
        print(f"labels saved to {fn}")

    # save colorized mesh
    save_colorized_mesh(mesh, labels, f'data/labeled/{ply_file[:-4]}_labeled.ply', colormap='scannet200')


def run():
    model = get_model('checkpoints/scannet200/scannet200_benchmark.ckpt')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    run_for_scene(model, "akitchen.ply")
    run_for_scene(model, "aliving.ply")
    run_for_scene(model, "bed.ply")
    run_for_scene(model, "kitchen.ply")
    run_for_scene(model, "mliving.ply")
    run_for_scene(model, "luke.ply")
    run_for_scene(model, "5a.ply")
    run_for_scene(model, "5b.ply")


if __name__ == "__main__":
    run()