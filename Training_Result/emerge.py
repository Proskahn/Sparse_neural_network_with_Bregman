import matplotlib.pyplot as plt
import os
import csv

def read_csv_to_dict(file_path):
    data = {}
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            epoch = int(row[0])
            value = float(row[1])
            data[epoch] = value
    return data

def plot_metric(ax, save_path, metric_file, label):
    file_path = os.path.join(save_path, metric_file)
    data = read_csv_to_dict(file_path)
    
    epochs = list(data.keys())
    values = list(data.values())
    
    ax.plot(epochs, values, marker='o', linestyle='-', label=label)
    ax.tick_params(axis='both', labelsize=14)
    return epochs

def create_and_display_plots_1x4(base_paths):
    fig, axs = plt.subplots(1, 4, figsize=(32, 8))  # 1 row, 4 columns
    titles = ['Validation Loss -λ₁ = 0.001', 'Sparsity - λ₁ =0.001', 'Validation Loss - λ₁ = 0.1', 'Sparsity - λ₁ = 0.1']
    ylabels = ['Loss', 'Non-zero entries', 'Loss', 'Non-zero entries']
    metric_files = ['validation_loss.csv', 'net_sparsity.csv']
    
    legends_1, legends_2 = [], []
    handles_1, handles_2 = [], []
    
    for i, base_path in enumerate(base_paths):
        subdirectories = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
        
        for ax, metric_file, title, ylabel in zip(axs[i*2:(i+1)*2], metric_files, titles[i*2:(i+1)*2], ylabels[i*2:(i+1)*2]):
            all_epochs = []
            for subdir in subdirectories:
                subdir_path = os.path.join(base_path, subdir)
                params = subdir.split('_')
                formatted_lr = params[1]
                formatted_lamda0 = params[3]
                formatted_lamda1 = params[5]
                formatted_opt = params[6]
                
                label = f'LR={formatted_lr}, λ₀={formatted_lamda0}, λ₁={formatted_lamda1}, OPT={formatted_opt}'
                epochs = plot_metric(ax, subdir_path, metric_file, label)
                all_epochs.extend(epochs)
                
                if i == 0 and label not in legends_1:
                    legends_1.append(label)
                    handles_1, _ = ax.get_legend_handles_labels()
                elif i == 1 and label not in legends_2:
                    legends_2.append(label)
                    handles_2, _ = ax.get_legend_handles_labels()
                
            ax.set_title(title, fontsize=25)
            ax.set_ylabel(ylabel, fontsize=25)
            ax.set_xlabel('Epoch', fontsize=25)
            ax.grid(True)
            
            if all_epochs:
                max_epoch = max(all_epochs)
                ax.set_xticks(range(0, max_epoch + 1, 10))
                ax.set_xticklabels(range(0, max_epoch + 1, 10))
    
    # Add two separate legends
    fig.legend(handles_1, legends_1, loc='upper left', fontsize=18, ncol=1, bbox_to_anchor=(0.3, 0.6))
    fig.legend(handles_2, legends_2, loc='upper right', fontsize=18, ncol=1, bbox_to_anchor=(1.0, 0.6))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate legends
    plt.savefig('Training_Result/Figure/Figure_combined')
    plt.show()

# Example usage
base_paths = ['Training_Result/Figure/Figure3', 'Training_Result/Figure/Figure4']
create_and_display_plots_1x4(base_paths)
