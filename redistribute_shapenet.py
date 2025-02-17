import os
import shutil
import zipfile

NUM_FOLDERS = 10

def empty_tmp():
    print("Emptying tmp folder...")
    shutil.rmtree("tmp", ignore_errors=True)

if __name__ == "__main__":
    os.makedirs(os.path.join("data","ShapenetRedistributed"), exist_ok=True)
    for i in range(NUM_FOLDERS):
        os.makedirs(os.path.join("data","ShapenetRedistributed",str(i)), exist_ok=True)

    for compressed in os.listdir(os.path.join("data","ShapeNetCore")):
        if compressed.endswith(".zip"):
            empty_tmp()
            with zipfile.ZipFile(os.path.join("data","ShapeNetCore",compressed), 'r') as zip_ref:
                print(f"Extracting {compressed}...")
                zip_ref.extractall("tmp")
                print(f"Extracted {compressed}")
                for root,dirs,files in os.walk("tmp"):
                    for file in files:
                        if file.endswith(".obj"):
                            print(root)
                            shutil.move(os.path.join(root,file), os.path.join("data","ShapenetRedistributed",str(hash(root)%NUM_FOLDERS),str(hash(root))+".obj"))


    empty_tmp()

    for i in range(NUM_FOLDERS):
        print(f"Folder {i} contains {len(os.listdir(os.path.join('data','ShapenetRedistributed',str(i))))} files")
        #zip the folder
        shutil.make_archive(os.path.join("data","ShapenetRedistributed",str(i)), 'zip', os.path.join("data","ShapenetRedistributed",str(i)))
        #remove the folder
        shutil.rmtree(os.path.join("data","ShapenetRedistributed",str(i)), ignore_errors=True)
        

