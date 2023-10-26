from plyer import notification
from sklearn import datasets

# 下载LFW数据集到指定文件夹
lfw = datasets.fetch_lfw_people(download_if_missing=True, data_home="E:\\homework_machine_learning\\test4-SVM\\Iris data set\\LFW", funneled=False, resize=0.4)

# 下载完成后弹出通知
if lfw:
    notification.title = "下载完成"
    notification.message = "LFW dataset 下载已完成"
    notification.notify()
