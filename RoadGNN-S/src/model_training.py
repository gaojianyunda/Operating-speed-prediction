from data_processing import GraphDataProcessor
from GNNmain import GNNModel
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
from Knowledge_enhancement import compute_knowledge_enhanced_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    csv_file_path = 'default_path.csv'
    data_processor = GraphDataProcessor(csv_file_path)
    graphs, target_values = data_processor.process_data() 

    num_graphs = len(graphs)
    train_size = int(0.8 * num_graphs)
    batch_size = 8  # Set batch size

    train_graphs = graphs[:train_size]
    test_graphs = graphs[train_size:]
    train_true = target_values[:train_size]
    test_true = target_values[train_size:]

    model = GNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 200

    for epoch in range(num_epochs):
        ## Training mode
        model.train()
        train_rmse, train_mae, train_r2, train_mape = [], [], [], []

        for i in range(0, len(train_graphs), batch_size):
            batch_graphs = train_graphs[i:i + batch_size]
            batch_true = train_true[i:i + batch_size]
            
            optimizer.zero_grad()
            batch_graphs = Batch.from_data_list(batch_graphs)  
            batch_true = torch.FloatTensor(batch_true).to(device)  
            batch_graphs = batch_graphs.to(device)  
            
            out = model(batch_graphs)

            predicted_features = out[0]  

            if predicted_features.shape[0] != batch_true.shape[0]:
                predicted_features = predicted_features.view(batch_true.shape[0], -1)

            if predicted_features.shape[0] != batch_true.shape[0]:
                raise ValueError(f"Mismatch in number of samples: {predicted_features.shape[0]} vs {batch_true.shape[0]}")

            loss_data = nn.functional.mse_loss(predicted_features, batch_true)

            y_q_pred = predicted_features  
            y_q_true = batch_true
              
            ##  The geometric design parameters and the corresponding station numbers for the operating speed calculation need to be read from graphs.
       
            L_knwl = compute_knowledge_enhanced_loss(predicted_features, 
                                          R_H=1000.0, 
                                          f_max=0.5, 
                                          e_s=2.0, 
                                          b=2.0, 
                                          h_g=0.5, 
                                          vS_prev_sum=1.0, 
                                          vS_curr_sum=1.0, 
                                          V_85=30.0, 
                                          V_e=40.0, 
                                          V_dc_a_vn=1.0, 
                                          kk=0.01, 
                                          device=device)            

            total_loss = loss_data + L_knwl

            rmse = mean_squared_error(batch_true.detach().cpu().numpy(), predicted_features.detach().cpu().numpy(), squared=False)
            mae = mean_absolute_error(batch_true.detach().cpu().numpy(), predicted_features.detach().cpu().numpy())
            r2 = r2_score(batch_true.detach().cpu().numpy(), predicted_features.detach().cpu().numpy())
            mape = mean_absolute_percentage_error(batch_true.detach().cpu().numpy(), predicted_features.detach().cpu().numpy())

            train_rmse.append(rmse)
            train_mae.append(mae)
            train_r2.append(r2)
            train_mape.append(mape)

            total_loss.backward()
            optimizer.step()

        avg_rmse_T = np.mean(train_rmse)
        avg_mae_T = np.mean(train_mae)
        avg_r2_T = np.max(train_r2)
        avg_mape_T = np.mean(train_mape)

        ## Testing mode
        model.eval()
        test_rmse, test_mae, test_r2, test_mape = [], [], [], []

        for i in range(0, len(test_graphs), batch_size):
            batch_graphs = test_graphs[i:i + batch_size]
            batch_true = test_true[i:i + batch_size]
            with torch.no_grad():
                batch_graphs = Batch.from_data_list(batch_graphs)  # Create a batch from the list
                batch_true = torch.FloatTensor(batch_true).to(device)  # Move target values to GPU
                batch_graphs = batch_graphs.to(device)  # Move batch to GPU
                
                out = model(batch_graphs)
                predicted_features = out[0]

                # Calculate metrics
                rmse = mean_squared_error(batch_true.detach().cpu().numpy(), predicted_features.detach().cpu().numpy(), squared=False)
                mae = mean_absolute_error(batch_true.detach().cpu().numpy(), predicted_features.detach().cpu().numpy())
                r2 = r2_score(batch_true.detach().cpu().numpy(), predicted_features.detach().cpu().numpy())
                mape = mean_absolute_percentage_error(batch_true.detach().cpu().numpy(), predicted_features.detach().cpu().numpy())

                # Append metrics to lists
                test_rmse.append(rmse)
                test_mae.append(mae)
                test_r2.append(r2)
                test_mape.append(mape)

        # Display average training metrics
        print(f"train-Epoch {epoch + 1}: Average RMSE: {avg_rmse_T}, Average MAE: {avg_mae_T}, Average R2: {avg_r2_T}, Average MAPE: {avg_mape_T}")

        # Calculate average test metrics
        avg_rmse = np.mean(test_rmse)
        avg_mae = np.mean(test_mae)
        avg_r2 = np.mean(test_r2)
        avg_mape = np.mean(test_mape)

        # Display average test metrics
        print(f"test-Epoch {epoch + 1}: Average RMSE: {avg_rmse}, Average MAE: {avg_mae}, Average R2: {avg_r2}, Average MAPE: {avg_mape}")

    # Save the trained model
    torch.save(model.state_dict(), 'output/road_gnn_model.pth')
    print("Model saved to road_gnn_model.pth")
    
    # Example testing
    model.eval()
    data_batch = Batch.from_data_list([graphs[0]]).to(device)  
    output = model(data_batch)
    print("True value:", target_values[0])
    print("Predicted value:", output[0].detach().cpu().numpy())

if __name__ == "__main__":
    main()
