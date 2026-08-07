loss_function2=zeros(6,length(repeat_times));
computation_time_function2=zeros(6,length(repeat_times));
phase1_budget_function2=zeros(6,length(repeat_times));
safety_margin_function2=zeros(6,length(repeat_times));
divide_parameter=[80,90,100,110,120,125,130,140,150,160,170,180,190,200,210,220,230,240];
for id_num_rec=1:1
    di_para=divide_parameter(1); % 80
    for id_repeat_times=1:repeat_times
        main_BD_function2_rome;
        loss_function2(id_num_rec,id_repeat_times)=loss;
        computation_time_function2(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function2(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function2(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=2:2
    di_para=divide_parameter(4); % 110
    for id_repeat_times=1:repeat_times
        main_BD_function2_rome;
        loss_function2(id_num_rec,id_repeat_times)=loss;
        computation_time_function2(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function2(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function2(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=3:3
    di_para=divide_parameter(8); % 140
    for id_repeat_times=1:repeat_times
        main_BD_function2_rome;
        loss_function2(id_num_rec,id_repeat_times)=loss;
        computation_time_function2(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function2(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function2(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=4:4
    di_para=divide_parameter(10); % 160
    for id_repeat_times=1:repeat_times
        main_BD_function2_rome;
        loss_function2(id_num_rec,id_repeat_times)=loss;
        computation_time_function2(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function2(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function2(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=5:5
    di_para=divide_parameter(12); % 180
    for id_repeat_times=1:repeat_times
        main_BD_function2_rome;
        loss_function2(id_num_rec,id_repeat_times)=loss;
        computation_time_function2(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function2(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function2(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=6:6
    di_para=divide_parameter(16); % 220
    for id_repeat_times=1:repeat_times
        main_BD_function2_rome;
        loss_function2(id_num_rec,id_repeat_times)=loss;
        computation_time_function2(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function2(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function2(id_num_rec,id_repeat_times)=safety_margin;
    end
end
loss_function2_mean=mean(loss_function2, 2)
computation_time_function2_mean=mean(computation_time_function2,2);
PANDAp_vio=zeros(length(num_rec),1);
phase1_budget_function2_mean=mean(phase1_budget_function2(:));
safety_margin_function2_mean=mean(safety_margin_function2(:));
