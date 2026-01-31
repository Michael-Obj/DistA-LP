function [ineq_iis,eq_iis,ncalls,fval0] = elasticfilter(Aineq,bineq,Aeq,beq,lb,ub)
    ncalls = 0;
    [mi,n] = size(Aineq); % Number of variables and linear inequality constraints
    me = size(Aeq,1);
    Aineq_r = [Aineq -1.0*eye(mi) zeros(mi,2*me)];
    Aeq_r = [Aeq zeros(me,mi) eye(me) -1.0*eye(me)]; % Two slacks for each equality constraint
    lb_r = [lb(:); zeros(mi+2*me,1)];
    ub_r = [ub(:); inf(mi+2*me,1)];
    ineq_slack_offset = n;
    eq_pos_slack_offset = n + mi;
    eq_neg_slack_offset = n + mi + me;
    f = [zeros(1,n) ones(1,mi+2*me)];
    opts = optimoptions("linprog","Algorithm","dual-simplex","Display","none");
    tol = 1e-10;
    
    ineq_iis = false(mi,1);
    eq_iis = false(me,1);
    [x,fval,exitflag] = linprog(f,Aineq_r,bineq,Aeq_r,beq,lb_r,ub_r,[],opts);
    fval0 = fval;
    ncalls = ncalls + 1;
    while exitflag == 1 && fval > tol % Feasible and some slacks are nonzero
        c = 0;
        for i = 1:mi        
            j = ineq_slack_offset+i;
            if x(j) > tol
                ub_r(j) = 0.0;
                ineq_iis(i) = true;
                c = c+1;
            end
        end
        for i = 1:me
            j = eq_pos_slack_offset+i;
            if x(j) > tol            
                ub_r(j) = 0.0;
                eq_iis(i) = true;
                c = c+1;
            end
        end
        for i = 1:me
            j = eq_neg_slack_offset+i;
            if x(j) > tol            
                ub_r(j) = 0.0;
                eq_iis(i) = true;
                c = c+1;
            end
        end
        [x,fval,exitflag] = linprog(f,Aineq_r,bineq,Aeq_r,beq,lb_r,ub_r,[],opts);
        if fval > 0
            fval0 = fval;
        end
        ncalls = ncalls + 1;
    end 
end