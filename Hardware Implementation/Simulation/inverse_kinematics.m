function [theta1d,theta2d]=inverse_kinematics(xd,yd)
l1=0.2;
l2=0.15;
theta2d=acos((xd^2+yd^2-l1^2-l2^2)/(2*l1*l2));
theta1d=atan(yd/xd)-atan((l2*sin(theta2d))/(l1+l2*cos(theta2d)));
