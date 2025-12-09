from torch.utils.data import Dataset,DataLoader;
from PIL import Image;
from pathlib import Path;
from torchvision import transforms;


def text2logs(file_path,text = ""):
    with open(file_path,'w',encoding="utf-8") as f:
        f.write(text);

normalized_dict = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

# Define data transforms
def get_transforms(IMG_SIZE,jitter):
    print("Use Jitter")
    if jitter or jitter == "On":
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((IMG_SIZE,IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1) 
            ),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.4
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalized_dict['mean'], std=normalized_dict['std'])
        ])
    else:
        print("no jitter")
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((IMG_SIZE,IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=0,  # no rotation here (already done above)
                translate=(0.1, 0.1),  # shift up to 10% horizontally/vertically
                scale=(0.9, 1.1)  # scale alteration
            ),
            #transforms.RandomRotation(degrees=10),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalized_dict['mean'], std=normalized_dict['std'])
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalized_dict['mean'], std=normalized_dict['std'])
    ])
    
    return train_transform, test_transform

# Custom Dataset Class for Corn Disease Images
class CornDiseaseDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None,rmbg = False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.rmbg = rmbg;
        if self.rmbg:
            print("rmbg mode")
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx];
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.rmbg:
            img = Image.open(self.image_paths[idx]).convert("RGBA");
            background = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img = Image.alpha_composite(background.convert("RGBA"), img)
            image = rgb_img.convert("RGB");
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    

def make_dataloaders(train_path,
                     val_path,
                     selected_classes = set(),
                     img_size = 448,
                     batch_size = 32,
                     num_workers = 0,
                     log_path : str = "./",
                     jitter=False,
                     rmbg = False):
    
    assert  Path(train_path).is_dir() and \
            Path(val_path).is_dir(), \
            "... train/val path is not defined";
    assert len(selected_classes) > 0, "we found 0 classe";

    train_path = Path(train_path);
    val_path = Path(val_path);
    X_train_paths, y_train_labels, X_val_paths, y_val_labels = [],[],[],[];
    count = dict();
    class2id = dict();

    label = 0;
    prev_disease_name = None;
    for subs in zip(train_path.iterdir(),val_path.iterdir()):
        sub_train,sub_val = subs;
        
        train_disease_name = str(sub_train.name).split("_")[0].lower();
        val_disease_name = str(sub_val.name).split("_")[0].lower();

        assert train_disease_name == val_disease_name, "bug : name not sorted"
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"];

        if train_disease_name in selected_classes:
            
            train_paths = [e for e in sub_train.iterdir() if e.suffix in image_extensions];
            val_paths = [e for e in sub_val.iterdir() if e.suffix in image_extensions];
            count[f"{train_disease_name}_train"] = len(train_paths);
            count[f"{train_disease_name}_val"] = len(val_paths);

            X_train_paths.extend(train_paths);
            y_train_labels.extend([label] * len(train_paths));
            X_val_paths.extend(val_paths);
            y_val_labels.extend([label]*len(val_paths));

            if not train_disease_name in class2id:
                class2id[train_disease_name] = label;
            
            if prev_disease_name == None or prev_disease_name != train_disease_name:
                prev_disease_name = train_disease_name;
                label += 1;
    assert len(X_train_paths) == len(y_train_labels) and len(X_val_paths) == len(y_val_labels), "Bug! something went wrong"
    
    logs = "";
    train_cnt = 0;
    val_cnt = 0;
    for k in count:
        if "_train" in k:
            k = k.split("_")[0]
            logs += f"{k} | train : {count[f'{k}_train']} | val : {count[f'{k}_val']}" + "\n";
            train_cnt += count[f'{k}_train'];
            val_cnt += count[f'{k}_val'];
    logs += f"#Train = {train_cnt} samples ; #Validation = {val_cnt} samples";
    logs = logs.strip();
    text2logs(Path(log_path) / "dist.txt",logs);

    train_tf, test_tf = get_transforms(img_size,jitter)
    train_dataset = CornDiseaseDataset(X_train_paths,y_train_labels,transform=train_tf,rmbg=rmbg);
    val_dataset = CornDiseaseDataset(X_val_paths,y_val_labels,transform=test_tf,rmbg=rmbg);

    train_dl = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers);
    val_dl = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers);

    return train_dl,val_dl,class2id;

