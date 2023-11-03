%Satyam is GAY%
Ts=0.001;
[DOF_Arm,ArmInfo]=importrobot('Arm1');

%%

%p
% 
% 
%plot(out.Time,out.Data(1,:))
%hold on
%plot(out.Time,out.Data(2,:))
%plot3(anglesolve.Data(1,:),anglesolve.Data(2,:),linspace(1,length(anglesolve.Data(1,:)),length(anglesolve.Data(1,:))))
figure(1);plot3(out.Data(1,:),out.Data(2,:),out.Data(3,:))
%z=-0.1;
%xe=interp2(X,Y,Z,Y,z);
%figure(2);plot(xe,Y);

%%
save rbTree DOF_arm