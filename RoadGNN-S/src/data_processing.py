import pandas as pd
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

class GraphDataProcessor:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, encoding='GBK', engine='python')
             
        self.node_mapping = {
            'StartPoint': 0,
            'SpiralF': 1,
            'Circularcurve': 2,
            'SpiralB': 3,
            'GradeF': 4,
            'Verticalcurve': 5,
            'GradeB': 6,
            'Visualcurve1': 7,
            'Visualcurve2': 8,
            'Visualcurve3': 9,
            'Visualcurve4': 10,
            'Visualcurve5': 11,
            'Spatialcurve1': 12,
            'Spatialcurve2': 13,
            'Spatialcurve3': 14,
            'Spatialcurve4': 15,
            'Spatialcurve5': 16,
            'Speed1': 17,
            'Speed2': 18,
            'Speed3': 19,
            'Speed4': 20,
            'Speed5': 21,
            'EndPoint': 22
        }
        self.connections = [
            ('StartPoint', 'SpiralF'),
            ('StartPoint', 'GradeF'),
            ('StartPoint', 'Visualcurve1'),
            ('StartPoint', 'Speed1'),   
            ('StartPoint', 'Spatialcurve1'),
            
            ('SpiralF', 'GradeF'),
            ('SpiralF', 'Circularcurve'),
            ('SpiralF', 'Visualcurve2'),
            ('SpiralF', 'Spatialcurve2'),
            ('SpiralF', 'Speed2'),
        
            ('Circularcurve', 'Verticalcurve'),
            ('Circularcurve', 'SpiralB'), 
            ('Circularcurve', 'GradeF'),
            ('Circularcurve', 'Visualcurve3'),
            ('Circularcurve', 'Spatialcurve3'),
            ('Circularcurve', 'Speed2'),
            ('Circularcurve', 'Speed3'),
            ('Circularcurve', 'Speed4'),
            ('Circularcurve', 'Speed5'),
            
            ('SpiralB', 'EndPoint'),
            ('SpiralB', 'GradeB'), 
            ('SpiralB', 'Visualcurve4'),
            ('SpiralB', 'Spatialcurve4'),
            ('SpiralB', 'Visualcurve5'),
            ('SpiralB', 'Spatialcurve5'),
            ('SpiralB', 'Speed4'),
            ('SpiralB', 'Speed5'),
            
            ('GradeF', 'Verticalcurve'),
            ('GradeF', 'Visualcurve2'), 
            ('GradeF', 'Spatialcurve2'),  
            ('GradeF', 'Speed2'), 
            
            ('Verticalcurve', 'GradeB'),
            ('Verticalcurve', 'Visualcurve3'), 
            ('Verticalcurve', 'Spatialcurve3'), 
            ('Verticalcurve', 'Speed2'),  
            ('Verticalcurve', 'Speed3'),  
            ('Verticalcurve', 'Speed4'),   
            ('Verticalcurve', 'Speed5'),                       
 
            ('GradeB', 'EndPoint'),
            ('GradeB', 'Visualcurve4'), 
            ('GradeB', 'Spatialcurve4'),  
            ('GradeB', 'Speed4'), 
            ('GradeB', 'Speed5'), 
                    
            ('Visualcurve1', 'Speed1'),
            ('Visualcurve1', 'Speed2'),  
            ('Visualcurve2', 'Speed2'),
            ('Visualcurve2', 'Speed3'),  
            ('Visualcurve3', 'Speed3'),
            ('Visualcurve3', 'Speed4'),
            ('Visualcurve4', 'Speed4'),
            ('Visualcurve4', 'Speed5'),
            ('Visualcurve5', 'Speed5'),
            ('Visualcurve5', 'EndPoint'),
            
            ('Visualcurve1', 'Visualcurve2'),    
            ('Visualcurve2', 'Visualcurve3'),
            ('Visualcurve3', 'Visualcurve4'),
            ('Visualcurve4', 'Visualcurve5'),      
            ('Visualcurve5', 'EndPoint'),  
            
            ('Spatialcurve1', 'Spatialcurve2'),    
            ('Spatialcurve2', 'Spatialcurve3'),
            ('Spatialcurve3', 'Spatialcurve4'),
            ('Spatialcurve4', 'Spatialcurve5'),      
            ('Spatialcurve5', 'EndPoint'),  
    
            ('Spatialcurve1', 'Speed1'),
            ('Spatialcurve2', 'Speed2'),  
            ('Spatialcurve3', 'Speed3'),
            ('Spatialcurve4', 'Speed4'),
            ('Spatialcurve5', 'Speed5'),
      
            ('Speed1', 'Speed2'),
            ('Speed2', 'Speed3'),
            ('Speed3', 'Speed4'),
            ('Speed4', 'Speed5'),
            ('Speed5', 'EndPoint'),   

        ]
    
    def process_data(self):
        graphs = []
        target_values = []  # To store target values
        
        for i in range(len(self.df)):
            # Extract node features
            StartPoint_feature = self.df[['StartPoint']].values[i]
            EndPoint_feature = self.df[['EndPoint']].values[i]
            Circularcurve_feature = self.df[['Cur_RH', 'Len_RH']].values[i]
            Verticalcurve_feature = self.df[['R_V', 'Len_RV']].values[i]
            
            SpiralF_feature = self.df[['Len_LS1', 'A_LS1']].values[i]
            SpiralB_feature = self.df[['Len_LS2', 'A_LS2']].values[i]
            GradeF_feature = self.df[['i_VF', 'Len_VF']].values[i]
            GradeB_feature = self.df[['i_VB', 'Len_VB']].values[i]            
                        
            Visualcurve_feature1 = self.df[['NLen_VS0', 'MLen_VS0', 'LLen_VS0']].values[i]
            Visualcurve_feature2 = self.df[['NLen_VS1', 'MLen_VS1', 'LLen_VS1']].values[i]
            Visualcurve_feature3 = self.df[['NLen_VS2', 'MLen_VS2', 'LLen_VS2']].values[i]
            Visualcurve_feature4 = self.df[['NLen_VS3', 'MLen_VS3', 'LLen_VS3']].values[i]
            Visualcurve_feature5 = self.df[['NLen_VS4', 'MLen_VS4', 'LLen_VS4']].values[i]
            
            Spatialcurve_feature1 = self.df[['f0_Spital', 'd0_Spital']].values[i]
            Spatialcurve_feature2 = self.df[['f1_Spital', 'd1_Spital']].values[i]
            Spatialcurve_feature3 = self.df[['f2_Spital', 'd2_Spital']].values[i]
            Spatialcurve_feature4 = self.df[['f3_Spital', 'd3_Spital']].values[i]
            Spatialcurve_feature5 = self.df[['f4_Spital', 'd4_Spital']].values[i]
            
            Speed_feature1 = self.df[['V0']].values[i]
            Speed_feature2 = self.df[['VV1']].values[i]
            Speed_feature3 = self.df[['VV2']].values[i]
            Speed_feature4 = self.df[['VV3']].values[i]
            Speed_feature5 = self.df[['VV4']].values[i]
            

           # Extract target values
            Speed_feature2_target = self.df[['V1']].values[i]
            Speed_feature3_target = self.df[['V2']].values[i]
            Speed_feature4_target = self.df[['V3']].values[i]
            Speed_feature5_target = self.df[['V4']].values[i]
            
            # Create node features tensor
            node_features = torch.tensor([
                [StartPoint_feature[0], 0], 
                [SpiralF_feature[0], SpiralF_feature[1]], 
                [Circularcurve_feature[0], Circularcurve_feature[1]], 
                [SpiralB_feature[0], SpiralB_feature[1]], 
                [GradeF_feature[0], GradeF_feature[1]],                 
                [Verticalcurve_feature[0], Verticalcurve_feature[1]],
                [GradeB_feature[0], GradeB_feature[1]],                                 
                [Visualcurve_feature1[0], Visualcurve_feature1[1]],
                [Visualcurve_feature2[0], Visualcurve_feature2[1]],
                [Visualcurve_feature3[0], Visualcurve_feature3[1]],
                [Visualcurve_feature4[0], Visualcurve_feature4[1]],
                [Visualcurve_feature5[0], Visualcurve_feature5[1]],
                [Spatialcurve_feature1[0], Spatialcurve_feature1[1]],
                [Spatialcurve_feature2[0], Spatialcurve_feature2[1]],
                [Spatialcurve_feature3[0], Spatialcurve_feature3[1]],
                [Spatialcurve_feature4[0], Spatialcurve_feature4[1]],
                [Spatialcurve_feature5[0], Spatialcurve_feature5[1]],
                [Speed_feature1[0], 0],
                [Speed_feature2[0], 0],
                [Speed_feature3[0], 0],
                [Speed_feature4[0], 0],
                [Speed_feature5[0], 0],
                [EndPoint_feature[0], 0]
            ], dtype=torch.float)
            
            # Build edge index
            edge_index = []
            for connection in self.connections:
                source_node = self.node_mapping[connection[0]]
                target_nodes = [self.node_mapping[node] for node in connection[1:]]  # Remove the first element (source node)
                for target_node in target_nodes:
                    edge_index.append([source_node, target_node])
    
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            Speed2_indices = 18
            Speed3_indices = 19
            Speed4_indices = 20
            Speed5_indices = 21
            
            # Create edge features tensor
            edge_features = []
            for connection in self.connections:
                source_node_type = self.node_mapping[connection[0]]
                target_node_types = [self.node_mapping[node] for node in connection[1:]]  # Remove the first element (source node)
                for target_node_type in target_node_types:
                    edge_feature = torch.tensor([0.5, 0.5])      
                    edge_features.append(edge_feature)

            # Convert edge features to tensor
            edge_features = torch.stack(edge_features, dim=1).t().contiguous()

            # Create graph data structure and add edge features  该data 就是 PyTorch Geometric 的 Data 对象
            graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
            graphs.append(graph)

            # Append the target values for the current graph,这是GNN网络的输出值，目标对比值
            target_values.append([Speed_feature2_target[0], Speed_feature3_target[0],Speed_feature4_target[0], Speed_feature5_target[0]])

       
        target_values = torch.tensor(target_values, dtype=torch.float)  # Convert to tensor
   
        return graphs, target_values
    
    def visualize_graph(self, graph_index):
        graph = graphs[graph_index]
        G = to_networkx(graph, to_undirected=True)

        pos = nx.spring_layout(G, seed=68)

        # Fixed node positions
        pos = {
            0: np.array([0.0217521, 0.2512502]),
            1: np.array([0.19081028, 0.076048]),
            2: np.array([-0.27230036, 0.15812941]),
            3: np.array([0.32667571, 0.6174867]),
            4: np.array([-0.18636089, -0.5188142]),
            5: np.array([-0.48825035, 0.51799844]),
            6: np.array([0.88872302, 0.43832228]),
            7: np.array([0.09389151, -1.]),
            8: np.array([-0.97720866, 0.1519841]),
            9: np.array([0.70133283, -0.05481942]),
            10: np.array([0.20624809, -0.39934376]),
            11: np.array([-0.49748775, -0.16676645]),
            12: np.array([-0.00782554, -0.07147529]),
            13: np.array([0.42667571, 0.6174867]),
            14: np.array([-0.38636089, -0.5188142]),
            15: np.array([-0.58825035, 0.51799844]),
            16: np.array([0.38872302, 0.43832228]),
            17: np.array([0.29389151, -1.]),
            18: np.array([-0.17720866, 0.1519841]),
            19: np.array([0.30133283, -0.05481942]),
            20: np.array([0.40624809, -0.39934376]),
            21: np.array([-0.29748775, -0.16676645]),
            22: np.array([-0.30782554, -0.07147529])
        }

        node_colors = {
            0: 'red',    # StartPoint            
            1: 'blue',   # Circularcurve
            2: 'blue',   # Circularcurve
            3: 'blue',   # Circularcurve
            4: 'blue',   # Circularcurve
            5: 'blue',   # Circularcurve
            6: 'blue',   # Circularcurve
            
            7: 'purple', # Visualcurve1            
            8: 'purple', # Visualcurve1
            9: 'purple', # Visualcurve1
            10: 'purple', # Visualcurve2
            11: 'purple',  # Visualcurve3
            
            12: 'yellow',   # Spatialcurve1
            13: 'yellow',   # Spatialcurve2           
            14: 'yellow',   # Spatialcurve1
            15: 'yellow',   # Spatialcurve2
            16: 'yellow', # Spatialcurve3
            
            17: 'lime',   # Speed1
            18: 'lime',  # Speed2
            19: 'lime',  # Speed3
            20: 'lime',  # Speed2
            21: 'lime',  # Speed3
            
            22: 'red'    # EndPoint
        }

        node_colors_list = [node_colors[i] for i in range(graph.num_nodes)]

        edge_color = 'gray'
        edge_width = 1.5
        edge_alpha = 0.6
        arrow_style = '-|>'

        nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, node_size=300)
        nx.draw_networkx_edges(G, pos, edgelist=graph.edge_index.T.tolist(), edge_color=edge_color,
                               width=edge_width, alpha=edge_alpha, arrows=True, arrowstyle=arrow_style)

        node_labels = {i: f'{i}' for i in range(graph.num_nodes)}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_family='Times New Roman')
        plt.title(f'Visualization of Graph {graph_index}')
        plt.axis('off')
        plt.show()

# The visualize_graph function
    def visualize_graph_2(self,graph):     
           
        G = nx.DiGraph()
        plt.figure(figsize=(16, 10))

        for i in range(graph.num_nodes):
            node_features = [int(feature) for feature in graph.x[i].tolist()]
            G.add_node(i, label=f'Node {i}\nFeature: {node_features}')

        for source, target in graph.edge_index.t().tolist():
            G.add_edge(source, target)

        pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=300)

        nx.draw_networkx_edges(G, pos, edge_color='gray', arrowsize=10)

        node_labels = {i: data['label'] for i, data in G.nodes.data()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

        edge_labels = {}
        for i, (source, target) in enumerate(graph.edge_index.t().tolist()):
            edge_labels[(source, target)] = f'Edge {i}: {graph.edge_attr[i].tolist()}'
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
        
        plt.title('Visualization of Graph')
        plt.axis('off')
        plt.show()



# Example usage:
if __name__ == "__main__":

    processor = GraphDataProcessor('default_path.csv')
    graphs, target_values = processor.process_data()
    print(len(graphs[0].x))
    print(graphs[0])
    print(graphs[3].x[18:22])
    print(graphs[4].edge_index)
    print(target_values[2])

    processor.visualize_graph(0)    
    index_to_visualize = 0
    processor.visualize_graph_2(graphs[index_to_visualize])

