function [quad_obs,quad_act,pend_obs,quad_s0,t,cost,stage_cost,terminal_cost] = extract_raw(M)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
quad_obs = M(:,1:9);
quad_act = M(:,10:13);
pend_obs = M(:,14:22);
quad_s0 = M(:,23:30);
t = M(:,31);
cost = M(:,32);
stage_cost = M(:,33);
terminal_cost = M(:,34);
end

