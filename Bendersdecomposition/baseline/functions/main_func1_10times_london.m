loss_function1=zeros(6,repeat_times);
computation_time_function1=zeros(6,repeat_times);
phase1_budget_function1=zeros(6,length(repeat_times));
safety_margin_function1=zeros(6,length(repeat_times));
divide_parameter=[100   110   120   130   140   150   160   170   180   190   200   210   220   230   240   250   260   270   280   290   300];
for id_num_rec=1:1
    di_para=divide_parameter(4); % 
    for id_repeat_times=1:repeat_times
        main_BD_function1_london;
        loss_function1(id_num_rec,id_repeat_times)=loss;
        computation_time_function1(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function1(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function1(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=2:2
    di_para=divide_parameter(8); % 
    for id_repeat_times=1:repeat_times
        main_BD_function1_london;
        loss_function1(id_num_rec,id_repeat_times)=loss;
        computation_time_function1(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function1(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function1(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=3:3
    di_para=divide_parameter(12); % 140
    for id_repeat_times=1:repeat_times
        main_BD_function1_london;
        loss_function1(id_num_rec,id_repeat_times)=loss;
        computation_time_function1(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function1(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function1(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=4:4
    di_para=divide_parameter(16); % 160
    for id_repeat_times=1:repeat_times
        main_BD_function1_london;
        loss_function1(id_num_rec,id_repeat_times)=loss;
        computation_time_function1(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function1(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function1(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=5:5
    di_para=divide_parameter(19); % 190
    for id_repeat_times=1:repeat_times
        main_BD_function1_london;
        loss_function1(id_num_rec,id_repeat_times)=loss;
        computation_time_function1(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function1(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function1(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=6:6
    di_para=divide_parameter(21); % 220
    for id_repeat_times=1:repeat_times
        main_BD_function1_london;
        loss_function1(id_num_rec,id_repeat_times)=loss;
        computation_time_function1(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function1(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function1(id_num_rec,id_repeat_times)=safety_margin;
    end
end
loss_function1_mean=mean(loss_function1, 2)
computation_time_function1_mean=mean(computation_time_function1,2);
phase1_budget_function1_mean=mean(phase1_budget_function1(:));
safety_margin_function1_mean=mean(safety_margin_function1(:));
