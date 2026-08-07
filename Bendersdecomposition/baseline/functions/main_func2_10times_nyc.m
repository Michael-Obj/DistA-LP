loss_function2=zeros(6,repeat_times);
computation_time_function2=zeros(6,repeat_times);
phase1_budget_function2=zeros(6,length(repeat_times));
safety_margin_function2=zeros(6,length(repeat_times));
divide_parameter=[300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500];
for id_num_rec=1:1
    di_para=divide_parameter(3); % 80
    for id_repeat_times=1:repeat_times
        main_BD_function2_nyc;
        loss_function2(id_num_rec,id_repeat_times)=loss;
        computation_time_function2(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function2(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function2(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=2:2
    di_para=divide_parameter(7); % 110
    for id_repeat_times=1:repeat_times
        main_BD_function2_nyc;
        loss_function2(id_num_rec,id_repeat_times)=loss;
        computation_time_function2(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function2(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function2(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=3:3
    di_para=divide_parameter(10); % 140
    for id_repeat_times=1:repeat_times
        main_BD_function2_nyc;
        loss_function2(id_num_rec,id_repeat_times)=loss;
        computation_time_function2(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function2(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function2(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=4:4
    di_para=divide_parameter(13); % 160
    for id_repeat_times=1:repeat_times
        main_BD_function2_nyc;
        loss_function2(id_num_rec,id_repeat_times)=loss;
        computation_time_function2(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function2(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function2(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=5:5
    di_para=divide_parameter(16); % 180
    for id_repeat_times=1:repeat_times
        main_BD_function2_nyc;
        loss_function2(id_num_rec,id_repeat_times)=loss;
        computation_time_function2(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function2(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function2(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=6:6
    di_para=divide_parameter(20); % 220
    for id_repeat_times=1:repeat_times
        main_BD_function2_nyc;
        loss_function2(id_num_rec,id_repeat_times)=loss;
        computation_time_function2(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function2(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function2(id_num_rec,id_repeat_times)=safety_margin;
    end
end
loss_function2_mean=mean(loss_function2, 2)
computation_time_function2_mean=mean(computation_time_function2,2);
phase1_budget_function2_mean=mean(phase1_budget_function2(:));
safety_margin_function2_mean=mean(safety_margin_function2(:));

