
picture_dir=/home/hl46161/brace_root/2019_labelled_image/*/images

picture_dir=/home/hl46161/brace_root/2019_binary_mask/*/


distance_dir=/home/hl46161/brace_root/2019_label_test

awk -F. '{print $1}' )

for sample in $(ls $picture_dir); do
 
       echo $sample 
       #mkdir $sample
       #mkdir $sample/images 
       #mkdir $sample/mask 
       #mv ./${sample}.jpg ./$sample/images 
       #mv ./${sample}_*.tiff ./$sample/mask
       #rm -r ./${sample}/images_resize
       #rm -r ./${sample}/patches_images_resize
       #rm -r ./${sample}/patches_masks_resize
       rm -r ./${sample}/patches_masks_resize_border_800
       rm -r ./${sample}/patches_images_resize_800
       #ls ./${sample}/patches_images -1 | wc -l
       #ls ./${sample}/patches_masks -1 | wc -l
done


for sample in $(ls $picture_dir| grep tiff | awk -F_ '{print $1_$2}'); do
 
       echo $sample 
       #mkdir $sample
       #mkdir $sample/images 
       #mkdir $sample/mask 
       #mv ./${sample}.jpg ./$sample/images 
       #mv ./${sample}_*.tiff ./$sample/mask
       #rm -r ./${sample}/patches_images
       #rm -r ./${sample}/patches_masks
       #ls ./${sample}/patches_images -1 | wc -l
       #ls ./${sample}/patches_masks -1 | wc -l
done




for sample in $(ls $picture_dir| grep jpg); do

       cp $picture_dir/$sample $distance_dir/
       
done

scp -r ./CoCo_test_val/ hl46161@xfer.gacrc.uga.edu:/scratch/hl46161/brace_root/


rm /home/hl46161/brace_root/2019_labelled_image/*/patches_masks


5 6 10 43 53 32 


maskrcnn.15174276.err

ls $picture_dir| grep jpg 

| awk -F_ '{print $1"_"$2}'

mkdir $sample
       mkdir $sample/images 
       mkdir $sample/mask 
       mv ./${sample}.jpg ./$sample/images 
       mv ./${sample}_*.tiff ./$sample/mask

       
       for sample in $(ls $picture_dir| grep jpg | awk -F. '{print $1}' ); do

       mkdir $sample
       mkdir $sample/images 
       mkdir $sample/mask 
       mv ./${sample}.jpg ./$sample/images 
       mv ./${sample}_*.tiff ./$sample/mask
       
done


distance_dir=/home/hl46161/brace_root/2019_labelled_image

picture_dir=/home/hl46161/brace_root/2019_labelled_image

ls $picture_dir| grep tiff | awk -F_ '{print $1"_"$2"_"$3}'

ls $picture_dir| grep tiff | awk -F_brace '{print $1}'

for sample in $(ls $picture_dir| grep tiff | awk -F_ '{print $1"_"$2"_"$3}' ); do

       echo $sample
       
       echo ${sample}_braceroot_stalk_border.ome.tiff
       #mkdir $sample/mask_with_border
       #rm -r $sample/mask_with_border
       #mv ./${sample}_braceroot_stalk_border.ome.tiff ./$sample/mask_with_border
       #cp /home/hl46161/brace_root/compiled_data/${sample}.jpg ./
       #mkdir $sample
       #mkdir $sample/images 
       #mkdir $sample/mask 
       #mv ./${sample}.jpg ./$sample/images 
       #mv ./${sample}_*.tiff ./$sample/mask
       
       
done

picture_dir=/home/hl46161/brace_root/2019_new_annotation/fran pt3 - export(2)



for sample in $(ls $picture_dir| grep tiff | awk -F.tiff '{print $1}'); do

       echo $sample
       
       #cp ${sample}.tiff ${sample}_1.tiff
       #echo ${sample}.jpg
       #cp /home/hl46161/brace_root/compiled_data/${sample}.jpg ./
       #mkdir $sample
       #mkdir $sample/images 
       #mkdir $sample/mask 
       #mv ./${sample}.jpg ./$sample/images 
       #mv ./${sample}_*.tiff ./$sample/mask
       
       
done


scp –r ./folder/ hl46161@xfer.gacrc.uga.edu:/scratch//workDir/

ssh hl46161@sapelo2.gacrc.uga.edu

Python/3.8.2


conda create -p /home/hl46161/python3_conda -c bioconda  Python=3.8.2

interact -c 3 --mem=200G --time=18:00:00


//wp-rifs03.msmyid.uga.edu/eddie

############################################################

picture_dir=/home/hl46161/brace_root/2019_annotation_remaster/2019_annotation_remaster_mask

for sample in $(ls |awk -F_ '{print $1"_"$2"_"$3}' | uniq); do
 
       echo $sample 
       #mkdir $sample
       #mkdir $sample/images 
       #mkdir $sample/mask 
       #rename file 
       #mv ./${sample}_Stalk.ome.tiff ./${sample}_stalk.ome.tiff
       #mv ./${sample}_braceroot.ome.tiff ./$sample/
       #mv ./${sample}_stalk.ome.tiff ./$sample/
       #mv ./${sample}_whitelabel.ome.tiff ./$sample/
       #mv ./$sample/${sample}_braceroot_stalk_border_whitelabel.ome.tiff ./$sample/mask
       #mv ${sample}_braceroot_stalk_border_whitelabel.ome.tiff ./$sample/mask
       #cp /home/hl46161/brace_root/compiled_data/${sample}.jpg ./$sample/images/
       #mv ./${sample}.jpg ./$sample/images 
       #mv ./${sample}_*.tiff ./$sample/mask
       rm ./${sample}/${sample}_stalk.ome.tiff
       rm ./${sample}/${sample}_braceroot.ome.tiff
       rm ./${sample}/${sample}_whitelabel.ome.tiff
       #rm -r ./${sample}/images_resize
       #rm -r ./${sample}/images_resize_800
       #rm -r ./${sample}/images_resize_800_flip
       #rm -r ./${sample}/images_resize_400
       #rm -r ./${sample}/mask_resize_400
       #rm -r ./${sample}/images_resize_400_flip
       #rm -r ./${sample}/patches_masks_resize_400
       #rm -r ./${sample}/patches_masks_resize_400_flip
       #rm -r ./${sample}/patches_images_resize_400
       #rm -r ./${sample}/patches_images_resize_400_flip
       #rm -r ./${sample}/patches_images_resize
       #rm -r ./${sample}/patches_masks_resize
       #rm -r ./${sample}/patches_masks_resize_border_800
       #rm -r ./${sample}/patches_images_resize_800
       #ls ./${sample}/patches_images -1 | wc -l
       #ls ./${sample}/patches_masks -1 | wc -l
done

/home/hl46161/brace_root/compiled_data/${sample}.jpg


ls $picture_dir | awk -F_ '{print $1"_"$2"_"$3}' >>

##################################################################


scp -r ./2019_annotation_remaster_mask/ hl46161@xfer.gacrc.uga.edu:/scratch/hl46161/brace_root/

######################################################################


1294_plant_5A
1293_plant_5A
1296_plant_12A no white label 
1296_plant_13B
1316_plant_5A no brace root 
1319_plant_2B
1325_plant_1A no brace root 
1331_plant_1A only stalk
1349_plant_6A pnly brace_root
1349_plant_9A no brace root 
1355_plant_4B no brace root 
1365_plant_2A no brace root 
1367_plant_6B no brace root
1367_plant_7B only stalk
1372_plant_2A only stalk
1375_plant_2B no brace root
1377_plant_1A no brace root
1383_plant_10B no_brace_root
1384_plant_14B no_brace_root
1388_plant_3B no_brace_root
1388_plant_4B no_brace_root
1388_plant_5B no_brace_root
1388_plant_10A no_brace_root
1389_plant_6A no_brace_root
1389_plant_9B no brace_root
1403_plant_4A no_brace_root 
1410_plant_8B no_brace_root
1418_plant_2B
1422_plant_5B
1422_plant_6B
1435_plant_4B
1439_plant_9A
1440_plant_6A
1440_plant_7A
1440_plant_9A
1443_plant_1B
/1415_plant_12A
/1415_plant_13A
1351_plant_3A

1442_plant_5B

