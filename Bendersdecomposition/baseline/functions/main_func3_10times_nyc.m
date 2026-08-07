loss_function3=zeros(6,repeat_times);
computation_time_function3=zeros(6,repeat_times);
phase1_budget_function3=zeros(6,length(repeat_times));
safety_margin_function3=zeros(6,length(repeat_times));
divide_parameter=[300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500];
for id_num_rec=1:1
    di_para=divide_parameter(3); % 80
    for id_repeat_times=1:repeat_times
        main_BD_function3_nyc;
        loss_function3(id_num_rec,id_repeat_times)=loss;
        computation_time_function3(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function3(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function3(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=2:2
    di_para=divide_parameter(7); % 110
    for id_repeat_times=1:repeat_times
        main_BD_function3_nyc;
        loss_function3(id_num_rec,id_repeat_times)=loss;
        computation_time_function3(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function3(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function3(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=3:3
    di_para=divide_parameter(11); % 140
    for id_repeat_times=1:repeat_times
        main_BD_function3_nyc;
        loss_function3(id_num_rec,id_repeat_times)=loss;
        computation_time_function3(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function3(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function3(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=4:4
    di_para=divide_parameter(15); % 160
    for id_repeat_times=1:repeat_times
        main_BD_function3_nyc;
        loss_function3(id_num_rec,id_repeat_times)=loss;
        computation_time_function3(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function3(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function3(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=5:5
    di_para=divide_parameter(18); % 180
    for id_repeat_times=1:repeat_times
        main_BD_function3_nyc;
        loss_function3(id_num_rec,id_repeat_times)=loss;
        computation_time_function3(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function3(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function3(id_num_rec,id_repeat_times)=safety_margin;
    end
end
for id_num_rec=6:6
    di_para=divide_parameter(21); % 220
    for id_repeat_times=1:repeat_times
        main_BD_function3_nyc;
        loss_function3(id_num_rec,id_repeat_times)=loss;
        computation_time_function3(id_num_rec,id_repeat_times)=time_2PPO;
        phase1_budget_function3(id_num_rec,id_repeat_times)=phase1_budget;
        safety_margin_function3(id_num_rec,id_repeat_times)=safety_margin;
    end
end
loss_function3_mean=mean(loss_function3, 2)
computation_time_function3_mean=mean(computation_time_function3,2);
phase1_budget_function3_mean=mean(phase1_budget_function3(:));
safety_margin_function3_mean=mean(safety_margin_function3(:));

