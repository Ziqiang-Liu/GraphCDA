import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

torch.backends.cudnn.enabled = False

class GCN(nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(GCN, self).__init__()
        self.args = args
        self.gcn_cir1_f = GCNConv(self.args.fcir, self.args.fcir)
        self.gcn_cir2_f = GCNConv(self.args.fcir, self.args.fcir)
    
        self.gcn_dis1_f = GCNConv(self.args.fdis, self.args.fdis)
        self.gcn_dis2_f = GCNConv(self.args.fdis, self.args.fdis)
        
        

        self.cnn_cir = nn.Conv1d(in_channels=self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fcir, 1),
                               stride=1,
                               bias=True)
        self.cnn_dis = nn.Conv1d(in_channels=self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fdis, 1),
                               stride=1,
                               bias=True)

        self.gat_cir1_f = GATConv(self.args.fcir, self.args.fcir,heads=4,concat=False,edge_dim=1)

        self.gat_dis1_f = GATConv(self.args.fdis, self.args.fdis,heads=4,concat=False,edge_dim=1)


    def forward(self, data):
        torch.manual_seed(1)
        x_cir = torch.randn(self.args.circRNA_number, self.args.fcir)
        x_dis = torch.randn(self.args.disease_number, self.args.fdis)

 
        x_cir_f1 = torch.relu(self.gcn_cir1_f(x_cir.cuda(), data['cc']['edges'].cuda(), data['cc']['data_matrix'][data['cc']['edges'][0], data['cc']['edges'][1]].cuda()))
        x_cir_att= torch.relu(self.gat_cir1_f(x_cir_f1,data['cc']['edges'].cuda(),data['cc']['data_matrix'][data['cc']['edges'][0], data['cc']['edges'][1]].cuda()))
        x_cir_f2 = torch.relu(self.gcn_cir2_f(x_cir_att, data['cc']['edges'].cuda(), data['cc']['data_matrix'][data['cc']['edges'][0], data['cc']['edges'][1]].cuda()))
       
        
       
        
        x_dis_f1 = torch.relu(self.gcn_dis1_f(x_dis.cuda(), data['dd']['edges'].cuda(), data['dd']['data_matrix'][data['dd']['edges'][0], data['dd']['edges'][1]].cuda()))
        x_dis_att =torch.relu(self.gat_dis1_f(x_dis_f1, data['dd']['edges'].cuda(),data['dd']['data_matrix'][data['dd']['edges'][0], data['dd']['edges'][1]].cuda()))        
        x_dis_f2 = torch.relu(self.gcn_dis2_f(x_dis_att, data['dd']['edges'].cuda(), data['dd']['data_matrix'][data['dd']['edges'][0], data['dd']['edges'][1]].cuda()))
        
        X_cir = torch.cat((x_cir_f1, x_cir_f2), 1).t()
        X_cir = X_cir.view(1, self.args.gcn_layers, self.args.fcir, -1)

        X_dis = torch.cat((x_dis_f1, x_dis_f2), 1).t()
        X_dis = X_dis.view(1, self.args.gcn_layers, self.args.fdis, -1)
    
        cir_fea = self.cnn_cir(X_cir)
        cir_fea = cir_fea.view(self.args.out_channels, self.args.circRNA_number).t()

        dis_fea = self.cnn_dis(X_dis)
        dis_fea = dis_fea.view(self.args.out_channels, self.args.disease_number).t()

        return cir_fea.mm(dis_fea.t()),cir_fea,dis_fea










