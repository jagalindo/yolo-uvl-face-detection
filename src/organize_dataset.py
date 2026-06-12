import os
import shutil
import zipfile

def organize_yolo_data():
    # الحصول على مسار سطح المكتب
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    
    # مسار المجلد الجديد الذي سيحتوي على البيانات المرتبة
    dataset_dir = os.path.join(desktop_path, 'My_YOLO_Dataset')
    
    images_train_dir = os.path.join(dataset_dir, 'images', 'train')
    images_val_dir = os.path.join(dataset_dir, 'images', 'val')
    labels_train_dir = os.path.join(dataset_dir, 'labels', 'train')
    labels_val_dir = os.path.join(dataset_dir, 'labels', 'val')
    
    # إنشاء المجلدات الجديدة
    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(d, exist_ok=True)
        
    classes_found = set()

    # دالة لفك الضغط إذا كانت الملفات مضغوطة
    def extract_if_needed(folder_name):
        folder_path = os.path.join(desktop_path, folder_name)
        zip_path = os.path.join(desktop_path, folder_name + '.zip')
        
        if not os.path.exists(folder_path) and os.path.exists(zip_path):
            print(f"جاري فك ضغط {folder_name}.zip ...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(desktop_path)
        return folder_path

    # دالة لنقل الملفات من مجلد إلى المجلدات المنظمة
    def process_folder(source_folder, img_dest, lbl_dest):
        if not os.path.exists(source_folder):
            print(f"تنبيه: المجلد {source_folder} غير موجود على سطح المكتب.")
            return

        print(f"جاري ترتيب الملفات في {source_folder} ...")
        for root, dirs, files in os.walk(source_folder):
            for filename in files:
                file_path = os.path.join(root, filename)
                ext = filename.lower().split('.')[-1]
                
                # نقل الصور
                if ext in ['jpg', 'jpeg', 'png', 'bmp']:
                    shutil.copy(file_path, os.path.join(img_dest, filename))
                # نقل ملفات النصوص (labels)
                elif ext == 'txt' and filename.lower() != 'classes.txt':
                    shutil.copy(file_path, os.path.join(lbl_dest, filename))
                    
                    # محاولة قراءة رقم الكلاس
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts and parts[0].isdigit():
                                    classes_found.add(int(parts[0]))
                    except Exception as e:
                        pass

    # تجهيز المسارات
    train_source = extract_if_needed('train')
    val_source = extract_if_needed('val')
    
    # معالجة المجلدات
    process_folder(train_source, images_train_dir, labels_train_dir)
    process_folder(val_source, images_val_dir, labels_val_dir)
    
    # إنشاء ملف data.yaml
    if classes_found:
        num_classes = max(classes_found) + 1
    else:
        num_classes = 1
        
    class_names = [f"'class_{i}'" for i in range(num_classes)]
    
    yaml_content = f"""path: {dataset_dir.replace(os.sep, '/')}
train: images/train
val: images/val

nc: {num_classes}
names: [{', '.join(class_names)}]
"""
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
        
    print("-" * 50)
    print("✅ تم الانتهاء بنجاح!")
    print(f"📁 تم إنشاء المجلد الجديد المنظم في:\\n{dataset_dir}")
    print(f"📄 وتم إنشاء ملف data.yaml الخاص به.")
    print("ملاحظة: يمكنك فتح ملف data.yaml وتغيير 'class_0' إلى الأسماء الحقيقية للأشياء في صورك.")
    print("-" * 50)

if __name__ == "__main__":
    organize_yolo_data()
