import shutil
import os

def copy_file_to_folder(source_file, destination_folder):
    # Check if the source file exists
    if not os.path.exists(source_file):
        print(f"Source file '{source_file}' does not exist")
        return
    
    # Check if the destination folder exists, create if it doesn't
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Get the base filename from the source file
    file_name = os.path.basename(source_file)
    
    # Construct the destination file path
    destination_file = os.path.join(destination_folder, file_name)
    
    try:
        # Copy the file
        shutil.copy(source_file, destination_file)
        print(f"File '{file_name}' successfully copied to '{destination_folder}'")
    except Exception as e:
        print(f"An error occurred while copying the file: {e}")

def move_folder_and_subfolders(source_folder, destination_folder):
    try:
        # Check if the destination folder exists, if not, create it
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Move the folder and its subfolders to the destination folder
        shutil.move(source_folder, destination_folder)
        print(f"Folder '{source_folder}' and its subfolders have been successfully moved to '{destination_folder}'")
    except Exception as e:
        print(f"Failed to move the folder: {e}")



if __name__ == "__main__":
    # Example usage
    # move_folder_and_subfolders('/amax/data/lishuaixin/ckpts/retroformer_multi_modal/retroformer/ckpts_enz/seq_enz/random_split_untyped',
    #                            '/amax/data/lishuaixin/ckpts/retroformer_multi_modal/retroformer/ckpts_enz/seq_enz/old')
    # move_folder_and_subfolders('/amax/data/lishuaixin/ckpts/retroformer_multi_modal/retroformer/ckpts_enz/seq_enz/thres_0.6_frac_0.4_untyped',
    #                            '/amax/data/lishuaixin/ckpts/retroformer_multi_modal/retroformer/ckpts_enz/seq_enz/old')
    # move_folder_and_subfolders('/amax/data/lishuaixin/ckpts/retroformer_multi_modal/retroformer/ckpts_enz/seq_enz/thres_0.4_frac_0.7_untyped',
    #                            '/amax/data/lishuaixin/ckpts/retroformer_multi_modal/retroformer/ckpts_enz/seq_enz/old')
    # move_folder_and_subfolders('/amax/data/lishuaixin/retro_NP/bionavi-NP/baseline/thres0.6_split/thres0.6_split/',
    #                            '/amax/data/lishuaixin/retro_NP/bionavi-NP/baseline/thres0.6_split')
    copy_file_to_folder('/amax/data/lishuaixin/ckpts/baseline/retroformer/ckpts/ckpts/random_split_untyped/model_260000.pt',
                        '/amax/data/lishuaixin/ckpts/baseline/retroformer/ckpts/ckpts/random_split_untyped/save')
