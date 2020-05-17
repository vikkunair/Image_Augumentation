listing = dir('/Users/vikasnair/Documents/Personal/Surrey_MSc/Image_Processing_and_Deep_Learning/Amber_Download/cars/car_ims');

for j = 3:length(listing)
    current_file_name = listing(j).name;
    %disp(current_file_name);
    full_path_current_image = strcat(listing(j).folder,'/',listing(j).name);
    
    %Reading current image
    img_in=double(imread(full_path_current_image))./255;
    
    H=size(img_in,1);  % height
    W=size(img_in,2);  % width
    
    %Random number generator between 1-7
    angle=randi([1 10],1);
    th=pi/angle;
    
    R=[cos(th) -sin(th) 0 ; ...
       sin(th)  cos(th) 0 ; ...
       0        0       1];
    
   
   sf=randi([1 2],1);
    S=[sf 0 0 ; ...
       0  sf 0 ; ...
       0        0       1];

    T=[1 0 -W/2 ; ...
       0 1 -H/2 ; ...
       0 0 1];
   
    M=inv(T)*S*R*T;
    
    corners=[1 1 H W ; 1 W W 1];
    corners(3,:)=1;
    warped_corners=M*corners;
    minx=min(warped_corners(1,:));
    maxx=max(warped_corners(1,:));
    miny=min(warped_corners(2,:));
    maxy=max(warped_corners(2,:));

    shift=[1 0 -minx ; 0 1 -miny ; 0 0 1];
    M=shift*M;
    newW=ceil(maxx-minx+1);
    newH=ceil(maxy-miny+1);

    img_out=zeros(newH,newW,3);
    M=inv(M)
    for x=1:newW
        for y=1:newH
        
            q = [x ; y ; 1];
            p = M* q;
        
            u = p(1)/p(3);
            v = p(2)/p(3);

            vertblend=u-floor(u);
            horizblend=v-floor(v);
        
            if (u>1 & u<=W-1 & v>1 & v<=H-1) 
                A=img_in(floor(v),floor(u),:);
                B=img_in(ceil(v),floor(u),:);
                C=img_in(floor(v),ceil(u),:);
                D=img_in(ceil(v),ceil(u),:);
           
                E=A*(1-vertblend) + C*vertblend;
                F=B*(1-vertblend) + D*vertblend;
           
                G=(1-horizblend)*E + horizblend*F;
                img_out(y,x,:)=G;

            end
        
        end
    end
    disp(j);
    imwrite(img_out,strcat('/Users/vikasnair/Documents/Personal/Surrey_MSc/Image_Processing_and_Deep_Learning/Amber_Download/Transformed_Images/Iteration_1/',listing(j).name,'_',num2str(angle),'_','.jpg'))
end