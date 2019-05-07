function [] = image_proc( infile )
  image_path = '/n/home06/pkrastev/Computer/User_Codes/Example_Recipes/test';
  cd (strcat(image_path,'/INPUT'));
  I1 = imread(strcat(infile,'.jpg'));  
  I2 = rgb2gray(I1);
  I3 = imadjust(I2);
  cd (strcat(image_path,'/OUTPUT'));
  imwrite(I3, strcat(infile,'_out.jpg'));
  cd(image_path);
end
