%% method to compute the optical flow of the saved frames
addpath("data/frames");

i1 = imread("frame100.png");
i2 = ir("frame101.png");


flow_func = opticalFlowFarneback("PyramidScale",0.5, "FilterSize", 15,"NeighborhoodSize",7);

r_flow = estimateFlow(flow_func,i2(:,:,1));
g_flow = estimateFlow(flow_func,i2(:,:,2));
b_flow = estimateFlow(flow_func,i2(:,:,3));
imshow(i1)
hold on
plot(r_flow,'DecimationFactor',[5 5],'ScaleFactor',5);
hold off
fi(r_flow.Magnitude);
% fi(g_flow.Magnitude);
% fi(b_flow.Magnitude);