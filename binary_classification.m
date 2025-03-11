function binary()
rng("default")
f = figure();
set(gcf,'name','Binary Classification','NumberTitle','off','color','w','units','normalized','WindowState', 'maximized');
set(1,'defaultfiguretoolbar','figure');

uicontrol('style','text','units','normalized','position',[0 0.97 0.1 0.03],'string','Prevalence (%)','fontsize',12);
prevalence_text = uicontrol('style','edit','units','normalized','position',[0.1 0.97 0.1 0.03],'String','20','fontsize',12,'callback',@prevalence_edit_callback);

uicontrol('style','text','units','normalized','position',[0 0.94 0.1 0.03],'string','Number of Samples','fontsize',12);
N_text = uicontrol('style','edit','units','normalized','position',[0.1 0.94 0.1 0.03],'String',5000,'fontsize',12,'callback',@N_edit_callback);

uicontrol('style','text','units','normalized','position',[0 0.88 0.1 0.03],'string','Cutoff Threshold','fontsize',12);
cutoff_threshold_edit = uicontrol('style','slider','units','normalized','position',[0.1 0.88 0.1 0.03],'Value',0.5,'Min',0,'Max',1,'fontsize',12,'callback',@cutoff_threshold_edit_callback);

uicontrol('style','pushbutton','units','normalized','position',[0 0.85 0.2 0.03],'string','Find Best Threshold','callback',@find_best_threshold_callback,'fontsize',12);

uicontrol('style','text','units','normalized','position',[0.2 0.97 0.05 0.03],'string','Mu 0','fontsize',12);
mu_0_edit = uicontrol('style','slider','units','normalized','position',[0.25 0.97 0.1 0.03],'Value',0.3,'Min',0,'Max',1,'fontsize',12,'callback',@mu_0_edit_callback);

uicontrol('style','text','units','normalized','position',[0.2 0.94 0.05 0.03],'string','Sigma 0','fontsize',12);
sigma_0_edit = uicontrol('style','slider','units','normalized','position',[0.25 0.94 0.1 0.03],'Value',0.2,'Min',0,'Max',1,'fontsize',12,'callback',@sigma_0_edit_callback);

uicontrol('style','text','units','normalized','position',[0.35 0.97 0.05 0.03],'string','Mu 1','fontsize',12);
mu_1_edit = uicontrol('style','slider','units','normalized','position',[0.4 0.97 0.1 0.03],'Value',0.7,'Min',0,'Max',1,'fontsize',12,'callback',@mu_1_edit_callback);

uicontrol('style','text','units','normalized','position',[0.35 0.94 0.05 0.03],'string','Sigma 1','fontsize',12);
sigma_1_edit = uicontrol('style','slider','units','normalized','position',[0.4 0.94 0.1 0.03],'Value',0.2,'Min',0,'Max',1,'fontsize',12,'callback',@sigma_1_edit_callback);

uicontrol('style','text','units','normalized','position',[0.8 0.95 0.05 0.05],'string','Confusion Matrix','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.85 0.95 0.05 0.05],'string','+GT','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.9 0.95 0.05 0.05],'string','-GT','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.95 0.95 0.05 0.05],'string','Total','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.8 0.9 0.05 0.05],'string','+Pred','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.8 0.85 0.05 0.05],'string','-Pred','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.8 0.8 0.05 0.05],'string','Total','fontsize',12);
cm_tp = uicontrol('style','text','units','normalized','position',[0.85,0.9,0.05,0.05],'string',0,'fontsize',12);
cm_fn = uicontrol('style','text','units','normalized','position',[0.85,0.85,0.05,0.05],'string',0,'fontsize',12);
cm_fp = uicontrol('style','text','units','normalized','position',[0.9,0.9,0.05,0.05],'string',0,'fontsize',12);
cm_tn = uicontrol('style','text','units','normalized','position',[0.9,0.85,0.05,0.05],'string',0,'fontsize',12);
pred_positive = uicontrol('style','text','units','normalized','position',[0.95,0.9,0.05,0.05],'string','pred +','fontsize',12);
pred_negative = uicontrol('style','text','units','normalized','position',[0.95,0.85,0.05,0.05],'string','pred -','fontsize',12);
gt_positive = uicontrol('style','text','units','normalized','position',[0.85,0.8,0.05,0.05],'string','GT +','fontsize',12);
gt_negative = uicontrol('style','text','units','normalized','position',[0.9,0.8,0.05,0.05],'string','GT -','fontsize',12);
total = uicontrol('style','text','units','normalized','position',[0.95,0.8,0.05,0.05],'string','Total','fontsize',12);

uicontrol('style','text','units','normalized','position',[0.7 0.95 0.05 0.05],'string','Sen','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.7 0.9 0.05 0.05],'string','Spe','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.7 0.85 0.05 0.05],'string','PPV','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.7 0.8 0.05 0.05],'string','NPV','fontsize',12);
sen_text = uicontrol('style','text','units','normalized','position',[0.75 0.95 0.05 0.05],'string','Sensitivity','fontsize',12);
spe_text = uicontrol('style','text','units','normalized','position',[0.75 0.9 0.05 0.05],'string','Specificity','fontsize',12);
ppv_text = uicontrol('style','text','units','normalized','position',[0.75 0.85 0.05 0.05],'string','PPV','fontsize',12);
npv_text = uicontrol('style','text','units','normalized','position',[0.75 0.8 0.05 0.05],'string','NPV','fontsize',12);

uicontrol('style','text','units','normalized','position',[0.6 0.95 0.05 0.05],'string','+LR','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.6 0.9 0.05 0.05],'string','-LR','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.6 0.85 0.05 0.05],'string','F1-Score','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.6 0.8 0.05 0.05],'string','Accuracy','fontsize',12);
plr_text = uicontrol('style','text','units','normalized','position',[0.65 0.95 0.05 0.05],'string','+LR','fontsize',12);
nlr_text = uicontrol('style','text','units','normalized','position',[0.65 0.9 0.05 0.05],'string','-LR','fontsize',12);
f1_text = uicontrol('style','text','units','normalized','position',[0.65 0.85 0.05 0.05],'string','F1','fontsize',12);
acc_text = uicontrol('style','text','units','normalized','position',[0.65 0.8 0.05 0.05],'string','Acc','fontsize',12);

uicontrol('style','text','units','normalized','position',[0.5 0.95 0.05 0.05],'string','MCC','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.5 0.9 0.05 0.05],'string','Kappa','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.5 0.85 0.05 0.05],'string','BCE','fontsize',12);
uicontrol('style','text','units','normalized','position',[0.5 0.8 0.05 0.05],'string','Brier Score','fontsize',12);
mcc_text = uicontrol('style','text','units','normalized','position',[0.55 0.95 0.05 0.05],'string','MCC','fontsize',12);
kappa_text = uicontrol('style','text','units','normalized','position',[0.55 0.9 0.05 0.05],'string','Kappa','fontsize',12);
bce_text = uicontrol('style','text','units','normalized','position',[0.55 0.85 0.05 0.05],'string','BCE','fontsize',12);
brier_text = uicontrol('style','text','units','normalized','position',[0.55 0.8 0.05 0.05],'string','Brier Score','fontsize',12);



table_data = uitable(f,'columnname',{'GT','Pred'},'units','normalized','position',[0 0 0.15 0.8],'fontsize',10);
table_data.ColumnWidth = {75};

uicontrol('style','pushbutton','units','normalized','position',[0 0.8 0.15 0.03],'string','Load Data','callback',@load_data_callback,'fontsize',12);

simulate_reader();
idx = find_best_threshold_callback;
     
    function simulate_reader()
        prevalence = round(str2double(prevalence_text.String));
        N = round(str2double(N_text.String));
        mu_0 = mu_0_edit.Value;
        mu_1 = mu_1_edit.Value;
        sigma_0 = sigma_0_edit.Value;
        sigma_1 = sigma_1_edit.Value;
        [x, y] = simulate_reader_gaussian(N, prevalence, mu_0, mu_1, sigma_0, sigma_1);
        table_data.Data = [x;y]';
    end

    function idx = find_best_threshold_callback(~,~,~)
        x = table_data.Data(:,1);
        y = table_data.Data(:,2); 
        thres_values = linspace(0,1,101);
        [sen, spe, ~, ~, ~, ~, ~, ~] = calculate_metrics(x,y,thres_values);
        idx = find_the_best_thres(sen,spe);  
        cutoff_threshold_edit.Value = thres_values(idx);
        plot_curves(idx)  
    end

    function cutoff_threshold_edit_callback(~,~,~)
        cutoff_threshold = cutoff_threshold_edit.Value;
        thres_values = linspace(0,1,101);
        [~,idx] = min(abs(thres_values-cutoff_threshold));
        plot_curves(idx)    
    end

    function prevalence_edit_callback(~,~,~)
        simulate_reader(); 
        plot_curves(idx);
    end

    function N_edit_callback(~,~,~)        
        simulate_reader(); 
        plot_curves(idx);    
    end

    function mu_0_edit_callback(~,~,~)
        simulate_reader(); 
        plot_curves(idx);    
    end

    function mu_1_edit_callback(~,~,~)
        simulate_reader(); 
        plot_curves(idx);   
    end

    function sigma_0_edit_callback(~,~,~)
        simulate_reader(); 
        plot_curves(idx);   
    end

    function sigma_1_edit_callback(~,~,~)
        simulate_reader(); 
        plot_curves(idx);   
    end

    function plot_curves(idx) 
        x = table_data.Data(:,1);
        y = table_data.Data(:,2);   
        mu_0 = mu_0_edit.Value;
        mu_1 = mu_1_edit.Value;
        N = round(str2double(N_text.String));
        prevalence = round(str2double(prevalence_text.String));
        thres_values = linspace(0,1,101);
        [sen, spe, ppv, npv, f1, acc, mcc, kappa] = calculate_metrics(x,y,thres_values);
        area_under_the_roc_curve = round(abs(trapz(1-spe,sen)),2); 
        area_under_the_pr_curve = round(abs(trapz(sen,ppv)),2); 
        [bins, pv_0, pv_1, counts_0, counts_1] = reliability_diagram(x,y);        

        t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'none');
        t.InnerPosition = [0.25, 0.15, 0.65, 0.6];
        
        nexttile;
        cla()
        hold on                
        h0 = histogram(y(x==0),100,'FaceColor','b','FaceAlpha', 0.5,'EdgeColor','None');
        h0_bins = h0.BinCounts;
        h0_bins = max(h0_bins);        
        h1 = histogram(y(x==1),100,'FaceColor','r','FaceAlpha', 0.5,'EdgeColor','None');
        h1_bins = h1.BinCounts;
        h1_bins = max(h1_bins);          
        line([thres_values(idx), thres_values(idx)],[0, max(h0_bins,h1_bins)+5],'color','k','linestyle','--','linewidth',1)
        line([mu_0, mu_0],[0, max(h0_bins,h1_bins)+5],'color','b','linestyle','--','linewidth',1)
        line([mu_1, mu_1],[0, max(h0_bins,h1_bins)+5],'color','r','linestyle','--','linewidth',1)
        xlabel('Probability','Interpreter','latex','fontsize',12)
        ylabel('Counts','Interpreter','latex','fontsize',12)
        title(strcat('N = ', string(N),', Prevalence = ', string(prevalence)),'Interpreter','latex','fontsize',12)
        set(gca,'TickLength',[0.02 0.02],'FontName','TimesNewRoman','FontSize',12,'TickLabelInterpreter','latex','box','on')
        xlim([0,1])
        ylim([0,max(h0_bins,h1_bins)+5])
        legend('Negative','Positive','FontSize',8)
        pbaspect([1,1,1])  

        nexttile;
        cla()
        hold on
        plot(thres_values, sen, 'Color','r','LineWidth',2)        
        plot(thres_values, spe, 'Color','b','LineWidth',2)
        line([thres_values(idx), thres_values(idx)],[0, 1],'color','k','linestyle','--','linewidth',1)
        xlabel('Probability Threshold','Interpreter','latex','fontsize',12)
        set(gca,'TickLength',[0.02 0.02],'FontName','TimesNewRoman','FontSize',12,'TickLabelInterpreter','latex','box','on')
        xlim([0,1])
        ylim([0,1])
        legend('TPR','FPR','FontSize',8)
        pbaspect([1,1,1])

        nexttile;
        cla()
        hold on
        plot(thres_values, ppv, 'Color','r','LineWidth',2)        
        plot(thres_values, npv, 'Color','b','LineWidth',2)
        line([thres_values(idx), thres_values(idx)],[0, 1],'color','k','linestyle','--','linewidth',1)
        xlabel('Probability Threshold','Interpreter','latex','fontsize',12)        
        set(gca,'TickLength',[0.02 0.02],'FontName','TimesNewRoman','FontSize',12,'TickLabelInterpreter','latex','box','on')
        xlim([0,1])
        ylim([0,1])
        legend('PPV','NPV','FontSize',8)
        pbaspect([1,1,1])

        nexttile;
        cla()
        hold on
        plot(thres_values, f1, 'Color','k','LineWidth',2) 
        plot(thres_values, acc, 'Color','m','LineWidth',2)
        plot(thres_values, mcc, 'Color','c','LineWidth',2)
        plot(thres_values, kappa, 'Color','g','LineWidth',2)
        line([thres_values(idx), thres_values(idx)],[0, 1],'color','k','linestyle','--','linewidth',1)
        xlabel('Probability Threshold','Interpreter','latex','fontsize',12)       
        set(gca,'TickLength',[0.02 0.02],'FontName','TimesNewRoman','FontSize',12,'TickLabelInterpreter','latex','box','on')
        legend('F1-Score','Accuracy','MCC','Cohen Kappa','FontSize',8,'Location','northeastoutside')
        pbaspect([1,1,1])

        nexttile;
        cla()
        hold on
        area(1-spe, sen,'FaceColor','k','FaceAlpha',0.3);
        scatter(1-spe, sen,3,'k')
        text(0.35,0.5,strcat('AUROC=',num2str(area_under_the_roc_curve)),'Interpreter','latex','FontName','TimesNewRoman','FontSize',12)
        scatter(1-spe(idx),sen(idx),'k')        
        text((1-spe(idx))+0.02, sen(idx)-0.04, strcat('Threshold = ',num2str(round(thres_values(idx),2))),'Interpreter','latex','FontName','TimesNewRoman','FontSize',12)
        line([1-spe(idx),1-spe(idx)],[0, 1],'color','k','linestyle','--','linewidth',1)        
        xlabel('FPR','Interpreter','latex','fontsize',12)
        ylabel('TPR','Interpreter','latex','fontsize',12)
        set(gca,'TickLength',[0.02 0.02],'FontName','TimesNewRoman','FontSize',12,'TickLabelInterpreter','latex','box','on')
        xlim([0,1])
        ylim([0,1])
        pbaspect([1,1,1])                

        nexttile;
        cla()
        hold on
        area(sen, ppv,'FaceColor','k','FaceAlpha',0.3);    
        scatter(sen, ppv,3,'k')
        text(0.35,0.5,strcat('AUC=',num2str(area_under_the_pr_curve)),'Interpreter','latex','FontName','TimesNewRoman','FontSize',12)
        line([sen(idx),sen(idx)],[0, 1],'color','k','linestyle','--','linewidth',1)        
        xlabel('Recall','Interpreter','latex','fontsize',12)
        ylabel('Precision','Interpreter','latex','fontsize',12)
        set(gca,'TickLength',[0.02 0.02],'FontName','TimesNewRoman','FontSize',12,'TickLabelInterpreter','latex','box','on')
        xlim([0,1])  
        ylim([0,1])
        pbaspect([1,1,1])        
        
        nexttile;
        cla();
        hold on;                   
        bar(bins, pv_1, 'FaceColor' , 'r', 'EdgeColor', 'r', 'LineWidth', 1,'FaceAlpha',0.5);         
        line([0,1],[0,1],'color','k','linestyle','--','linewidth',1)            
        title('Reliability Diagram, Class 1','interpreter','latex','fontsize',10)
        xlabel('Probability Threshold','Interpreter','latex','fontsize',12)
        ylabel('Predictive Value','Interpreter','latex','fontsize',12)   
        set(gca,'TickLength',[0.02 0.02],'FontName','TimesNewRoman','FontSize',12,'TickLabelInterpreter','latex','box','on')
        xlim([0,1])
        ylim([0,1])
        pbaspect([1,1,1])      

        nexttile;
        cla();
        hold on;
        bar(bins, counts_0+counts_1, 'FaceColor' , 'k', 'EdgeColor', 'k', 'LineWidth', 1,'FaceAlpha',0.5);                     
        title('Total Correct Predicitons','interpreter','latex','fontsize',10)                
        xlabel('Probability Threshold','Interpreter','latex','fontsize',12)
        ylabel('Counts','Interpreter','latex','fontsize',12)   
        set(gca,'TickLength',[0.02 0.02],'FontName','TimesNewRoman','FontSize',12,'TickLabelInterpreter','latex','box','on')
        xlim([0,1])
        pbaspect([1,1,1]) 

        [tp, fp, fn, tn] = calculate_tp_fp_fn_tn(x,y, thres_values(idx));        
        cm_tp.String = tp;
        cm_fn.String = fn;
        cm_fp.String = fp;
        cm_tn.String = tn;  
        pred_positive.String = tp+fp;
        pred_negative.String = fn+tn;
        gt_positive.String = tp+fn;
        gt_negative.String = fp+tn;
        total.String = tp+fp+fn+tn;
        sen_text.String = num2str(round(tp/(tp+fn),3));
        spe_text.String = num2str(round(tn/(tn+fp),3));        
        ppv_text.String = num2str(round((tp)/(fp+tp),3));
        npv_text.String = num2str(round((tn)/(fn+tn),3));
        plr_text.String = num2str(round(tp/(tp+fn),3)/(1-round(tn/(tn+fp),3)));
        nlr_text.String = num2str((1-round(tp/(tp+fn),3))/(round(tn/(tn+fp),3)));       
        f1_text.String = num2str(round((2*tp)/(2*tp+fp+fn),3));
        acc_text.String = num2str(round((tp+tn)/(tp+fn+fp+tn),3));
        mcc_text.String = num2str(round(((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),3));
        kappa_text.String = num2str(round(calculate_cohen_kappa(tp,fp,fn,tn),3));
        bce_text.String = num2str(round(calculate_bce(x,y),3));
        brier_text.String = num2str(round(calculate_brier(x,y),3));  
    end

    function load_data_callback(~,~,~)
        [file_name,path] = uigetfile('*.txt','Select .txt File','MultiSelect','off');
        if ~isequal(file_name,0)
            data = dlmread(fullfile(path,file_name));
            N_text.String = num2str(size(data,1));
            prevalence_text.String = num2str(sum(data(:,1))/size(data,1)*100);
            table_data.Data = data;
            idx = find_best_threshold_callback;
            plot_curves(idx)   
        end
    end
end

function [x, y] = simulate_reader_gaussian(N,prevalence,mu_0,mu_1,sigma_0,sigma_1)
number_of_positive = round(N*prevalence/100);
number_of_negative = N-number_of_positive;

pd_0 = makedist('Normal', 'mu', mu_0, 'sigma', sigma_0);
pd_0 = truncate(pd_0, 0, 1); 
y_0 = random(pd_0, number_of_negative, 1)';

pd_1 = makedist('Normal', 'mu', mu_1, 'sigma', sigma_1);
pd_1 = truncate(pd_1, 0, 1); 
y_1 = random(pd_1, number_of_positive, 1)';

x_0 = zeros(size(y_0));
x_1 = ones(size(y_1));
x = [x_0,x_1];
y = [y_0,y_1];
end

function [tp, fp, fn, tn] = calculate_tp_fp_fn_tn(x,y,threshold)
y_thres = y;
y_thres(y_thres<threshold) = 0;
y_thres(y_thres>=threshold) = 1;
tp = sum(and(y_thres==1, x==1));
fp = sum(and(y_thres==1, x==0));
tn = sum(and(y_thres==0, x==0));
fn = sum(and(y_thres==0, x==1));
end

function brier = calculate_brier(x,y)
brier = mean((y-x).^2);
end

function bcc = calculate_bce(x,y)
I = x==0;
bcc_0 = -log(1-y(I));
I = x==1;
bcc_1 = -log(y(I));
bcc = (mean(bcc_0)+mean(bcc_1))/2;
end

function kappa = calculate_cohen_kappa(tp,fp,fn,tn)
p_0 = (tp+tn)/(tp+fp+fn+tn);
p_A_yes = (tp+fp)/(tp+fp+fn+tn);
p_A_no = 1-p_A_yes;
p_B_yes = (tp+fn)/(tp+fp+fn+tn);
p_B_no = 1-p_B_yes;
p_yes = p_A_yes*p_B_yes;
p_no = p_A_no*p_B_no;
p_e = p_yes+p_no;
kappa = (p_0-p_e)/(1-p_e);
end

function [sen, spe, ppv, npv, f1, acc, mcc, kappa] = calculate_metrics(x,y,thres_values)
for i = 1:length(thres_values)
    [tp, fp, fn, tn] = calculate_tp_fp_fn_tn(x,y,thres_values(i));
    sen(i) = tp / (tp+fn);
    spe(i) = tn / (tn+fp);  
    ppv(i) = tp / (tp+fp);
    npv(i) = tn/ (tn+fn);
    f1(i) = 2*tp/(2*tp+fp+fn);   
    acc(i) = (tp+tn)/(tp+fp+fn+tn);
    mcc(i) = ((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));
    kappa(i) = calculate_cohen_kappa(tp,fp,fn,tn);
end 
ppv(isnan(ppv))=1;
end

function idx = find_the_best_thres(sen,spe)
spe = 1-spe;
dist = pdist2([0;1]',[spe;sen]');
[~,idx] = min(dist);
end

function [pv_0,counts_0] = calculate_class_0_tp_fp_fn_tn(x,y,threshold_1,threshold_2)
I = y<=threshold_2 & y>threshold_1;
x = x(I);
counts_0 = sum(x==0);
pv_0 = counts_0/length(x);
end

function [pv_1,counts_1] = calculate_class_1_tp_fp_fn_tn(x,y,threshold_1,threshold_2)
I = y<=threshold_2 & y>threshold_1;
x = x(I);
counts_1 = sum(x==1);
pv_1 = counts_1/length(x);
end

function [thres_values, pv_0, pv_1, counts_0, counts_1] = reliability_diagram(x,y)
thres_values_1 = 0:0.05:1;
thres_values_2 = 0.05:0.05:1.05;
for i = 1:length(thres_values_1)-1
    [pv_0(i),counts_0(i)] = calculate_class_0_tp_fp_fn_tn(x,y,thres_values_1(i),thres_values_2(i)); 
    [pv_1(i),counts_1(i)] = calculate_class_1_tp_fp_fn_tn(x,y,thres_values_1(i),thres_values_2(i)); 
end
thres_values = 0.025:0.05:1;
end