function [relevant_location_set, all_target] = get_relevant_location_set(q,user)
relevant_location_set=cell(length(q), 1);
random_matrix = rand(length(q),length(q));
positions = random_matrix < q;
for i=1:length(user)
    relevant_location_set{user(i)}=find(positions(user(i),:)==1);
end
all_target=[];
for i=1:length(user)
    all_target=[all_target, relevant_location_set{user(i)}];
end
all_target=unique(all_target);
% for i=1:length(user)
%     for j=1:length(user)
%         I=intersect(relevant_location_set{user(i)},relevant_location_set{user(j)});
%         if (i~=j)&&(~isempty(I))
% 
%             if ismember(user(i),I)
%                 relevant_location_set{user(j)}=setdiff(relevant_location_set{user(j)},user(i));
%                 I=setdiff(I,user(i));
%             end
% 
%             if ismember(user(j),I)
%                 relevant_location_set{user(i)}=setdiff(relevant_location_set{user(i)},user(j));
%                 I=setdiff(I,user(j));
%             end
% 
%             for length_I=1:length(I)
%                 lkrr=rand;
%                 if lkrr<length(relevant_location_set{user(i)})/(length(relevant_location_set{user(i)})+length(relevant_location_set{user(j)}))
%                     relevant_location_set{user(i)}=setdiff(relevant_location_set{user(i)},I(length_I));
%                 else
%                     relevant_location_set{user(j)}=setdiff(relevant_location_set{user(j)},I(length_I));
%                 end
% 
%             end
%     end
% end
end