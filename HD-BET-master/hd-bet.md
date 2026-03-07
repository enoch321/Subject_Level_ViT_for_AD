# NIfTI 文件在 nifti_output，要把去颅骨的文件存到 hdbet_output
hd-bet -i /root/autodl-tmp/NIfTI_output/ -o /root/autodl-tmp/hdbet_output/
# 在路径最前面加了 /（这叫绝对路径），Linux 系统的根目录往往是 /root/ 目录

在 HD-BET 跑完之后，在 Linux 终端进入你的 HD-BET 输出文件夹，运行下面这行命令，给所有文件加上 hdbet_ 前缀：
cd /root/autodl-tmp/hdbet_output/

# 批量给所有 .nii.gz 文件添加 hdbet_ 前缀（如果还没加的话）
for f in *.nii.gz; do mv "$f" "hdbet_$f"; done