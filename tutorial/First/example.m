x = [1; 2; 3];
y = [1; 2; 3];



params = -1:0.5:3; % change w from -1 to 3
curve = [];
for i = 1:length(params)
    w = params(i);
    h = w*x;
    
    subplot(2,1,1);
    plot(x,y,'rx'); 
    hold on
    plot(x,h,'b+-');
    axis([0 4 -4 10])
    xlabel('x','FontSize',20);
    ylabel('y','FontSize',20);
    str = ['w = ' num2str(params(i))];
    title(str,'FontSize',20);
    grid on;
    hold off;
    
    subplot(2,1,2);
    grid on;
    xlabel('w','FontSize',20);
    ylabel('L(w)','FontSize',20);
    title('Loss function','FontSize',20);
    hold on;
    axis([-1.2 3.2 -1 11]);
    L = lossFunctionL(h,y);
    curve = [curve [w;L]];
    plot(w,L,'mx');
    plot(curve(1,:),curve(2,:),'m-');
    
    pause(1);
end