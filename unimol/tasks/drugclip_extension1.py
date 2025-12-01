# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from IPython import embed as debug_embedded
import logging
import os
from collections.abc import Iterable
from sklearn.metrics import roc_auc_score
from xmlrpc.client import Boolean
import numpy as np
import torch
import pickle
from tqdm import tqdm
from unicore import checkpoint_utils
import unicore
from unicore.data import (AppendTokenDataset, Dictionary, EpochShuffleDataset,
                          FromNumpyDataset, NestedDictionaryDataset,
                          PrependTokenDataset, RawArrayDataset,LMDBDataset, RawLabelDataset,
                          RightPadDataset, RightPadDataset2D, TokenizeDataset,SortDataset,data_utils)
from unicore.tasks import UnicoreTask, register_task
from unimol.data import (AffinityDataset, CroppingPocketDataset,
                         CrossDistanceDataset, DistanceDataset,
                         EdgeTypeDataset, KeyDataset, LengthDataset,
                         NormalizeDataset, NormalizeDockingPoseDataset,
                         PrependAndAppend2DDataset, RemoveHydrogenDataset,
                         RemoveHydrogenPocketDataset, RightPadDatasetCoord,
                         RightPadDatasetCross2D, TTADockingPoseDataset, AffinityTestDataset, AffinityValidDataset, AffinityMolDataset, AffinityPocketDataset, ResamplingDataset)
#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve
logger = logging.getLogger(__name__)


def re_new(y_true, y_score, ratio):
    fp = 0
    tp = 0
    p = sum(y_true)
    n = len(y_true) - p
    num = ratio*n
    sort_index = np.argsort(y_score)[::-1]
    for i in range(len(sort_index)):
        index = sort_index[i]
        if y_true[index] == 1:
            tp += 1
        else:
            fp += 1
            if fp>= num:
                break
    return (tp*n)/(p*fp)


def calc_re(y_true, y_score, ratio_list):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    #print(fpr, tpr)
    res = {}
    res2 = {}
    total_active_compounds = sum(y_true)
    total_compounds = len(y_true)

    # for ratio in ratio_list:
    #     for i, t in enumerate(fpr):
    #         if t > ratio:
    #             #print(fpr[i], tpr[i])
    #             if fpr[i-1]==0:
    #                 res[str(ratio)]=tpr[i]/fpr[i]
    #             else:
    #                 res[str(ratio)]=tpr[i-1]/fpr[i-1]
    #             break
    
    for ratio in ratio_list:
        res2[str(ratio)] = re_new(y_true, y_score, ratio)

    #print(res)
    #print(res2)
    return res2

def cal_metrics(y_true, y_score, alpha):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """
    
        # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:,0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    count = 0
    # sort y_score, return index
    index  = np.argsort(y_score)[::-1]
    for i in range(int(len(index)*0.005)):
        if y_true[index[i]] == 1:
            count += 1
    auc = CalcAUC(scores, 1)
    ef_list = CalcEnrichment(scores, 1, [0.005, 0.01, 0.02, 0.05])
    ef = {
        "0.005": ef_list[0],
        "0.01": ef_list[1],
        "0.02": ef_list[2],
        "0.05": ef_list[3]
    }
    re_list = calc_re(y_true, y_score, [0.005, 0.01, 0.02, 0.05])
    return auc, bedroc, ef, re_list



@register_task("drugclip")
class DrugCLIP(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path",
        )
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--finetune-pocket-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=6.0,
            help="threshold for the distance between the molecule and the pocket",
        )
        parser.add_argument(
            "--max-pocket-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a pocket",
        )
        parser.add_argument(
            "--test-model",
            default=False,
            type=Boolean,
            help="whether test model",
        )
        parser.add_argument("--reg", action="store_true", help="regression task")

        parser.add_argument(
            "--atom-layer",
            type=int,
            default=-1,
            help="which transformer layer to use for atom embeddings; -1 = last, -2 = second-last, etc.",
        )

    def __init__(self, args, dictionary, pocket_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)
        self.mol_reps = None
        self.keys = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(args.data, "dict_mol.txt"))
        pocket_dictionary = Dictionary.load(os.path.join(args.data, "dict_pkt.txt"))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, mol_dictionary, pocket_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        data_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(data_path)
        if split.startswith("train"):
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")
            
            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
                True,
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
            
        else:
            
            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")


        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        mol_len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        pocket_len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                    "mol_len": RawArrayDataset(mol_len_dataset),
                    "pocket_len": RawArrayDataset(pocket_len_dataset)
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "pocket_name": RawArrayDataset(poc_dataset),
            },
        )
        if split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
            self.datasets[split] = ResamplingDataset(
                self.datasets[split]
            )
        else:
            self.datasets[split] = nest_dataset


    

    def load_mols_dataset(self, data_path,atoms,coords, **kwargs):
 
        dataset = LMDBDataset(data_path)
        label_dataset = KeyDataset(dataset, "label")
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )
        
        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "target":  RawArrayDataset(label_dataset),
                "mol_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset
    

    def load_retrieval_mols_dataset(self, data_path,atoms,coords, **kwargs):
 
        dataset = LMDBDataset(data_path)
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )
        
        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "mol_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset

    def load_pockets_dataset(self, data_path, **kwargs):

        dataset = LMDBDataset(data_path)
 
        dataset = AffinityPocketDataset(
            dataset,
            self.args.seed,
            "pocket_atoms",
            "pocket_coordinates",
            False,
            "pocket"
        )
        poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )




        apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")



        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                },
                "pocket_name": RawArrayDataset(poc_dataset),
                "pocket_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset

    

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        
        if args.finetune_mol_model is not None:
            print("load pretrain model weight from...", args.finetune_mol_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_mol_model,
            )
            model.mol_model.load_state_dict(state["model"], strict=False)
            
        if args.finetune_pocket_model is not None:
            print("load pretrain model weight from...", args.finetune_pocket_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_pocket_model,
            )
            model.pocket_model.load_state_dict(state["model"], strict=False)

        return model

    def train_step(
        self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *loss*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~unicore.data.UnicoreDataset`.
            model (~unicore.models.BaseUnicoreModel): the model
            loss (~unicore.losses.UnicoreLoss): the loss
            optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """

        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
    
    def valid_step(self, sample, model, loss, test=False):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = loss(model, sample)
        return loss, sample_size, logging_output


    def test_pcba_target(self, name, model, alpha=0.0, K=10, **kwargs):
        """Encode a dataset with the molecule encoder."""

        #names = "PPARG"
        data_path = "./data/lit_pcba/" + name + "/mols.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz=64
        print(num_data//bsz)
        mol_atom_list = []
        mol_global_list = []
        mol_names = []
        labels = []
        
        # generate mol data
        
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)

            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias, return_all_hidden_states=True
            )
            hidden_states = mol_outputs[-1]
            #print("num_hidden_states:", len(hidden_states))
            #for i, h in enumerate(hidden_states[-4:], start=len(hidden_states)-4):  # last 4
                #print("layer idx", i, "shape", h.shape)
            # quick peek of mean activations
            #print("last layer mean:", hidden_states[-1].mean().item())
            #print("second-last mean:", hidden_states[-2].mean().item())
            layer_idx = getattr(model.args if hasattr(model, "args") else model, "atom_layer", -1)
            chosen_hidden = hidden_states[layer_idx]  # [B, seq_len, D]
            # --- GLOBAL CLS embedding (keep existing pipeline) ---
            mol_cls = mol_outputs[0][:, 0, :]                      # [B, D]
            mol_global = model.mol_project(mol_cls)               # [B, Hg]
            mol_global = mol_global / (mol_global.norm(dim=-1, keepdim=True) + 1e-8)
            mol_global_np = mol_global.detach().cpu().numpy()
            mol_global_list.extend([mol_global_np[i] for i in range(mol_global_np.shape[0])])

            # --- PER-ATOM embeddings (exclude BOS/EOS tokens at positions 0 and -1) ---
            mol_atom_tensor = chosen_hidden[:, 1:-1, :].detach()  # [B, M_i, D]
            if mol_padding_mask is not None:
                mol_atom_mask = ~mol_padding_mask[:, 1:-1]
            else:
                mol_atom_mask = torch.ones(mol_atom_tensor.size(0), mol_atom_tensor.size(1), dtype=torch.bool, device=mol_atom_tensor.device)

            B, M_i, D = mol_atom_tensor.shape
            mol_atom_flat = mol_atom_tensor.reshape(-1, D)  # [B*M_i, D]
            mol_atom_proj_flat = model.mol_project(mol_atom_flat)  # [B*M_i, H]
            H = mol_atom_proj_flat.size(-1)
            mol_atom_proj = mol_atom_proj_flat.view(B, M_i, H)  # [B, M_i, H]
            mol_atom_proj = mol_atom_proj / (mol_atom_proj.norm(dim=-1, keepdim=True) + 1e-8)

            # Save each molecule's per-atom vectors (only valid atoms)
            for i in range(B):
                valid_len = int(mol_atom_mask[i].sum().item())
                if valid_len == 0:
                    mol_atom_list.append(np.zeros((0, H), dtype=np.float32))
                else:
                    arr = mol_atom_proj[i, :valid_len, :].detach().cpu().numpy()
                    mol_atom_list.append(arr)

            mol_names.extend(sample["smi_name"])
            labels.extend(sample["target"].detach().cpu().numpy())

        # Convert labels and global embeddings
        mol_global_arr = np.stack(mol_global_list, axis=0)  # [Nm, Hg]
        labels = np.array(labels, dtype=np.int32)
        # generate pocket data
        data_path = "./data/lit_pcba/" + name + "/pockets.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_atom_list = []
        pocket_global_list = []
        pocket_names = []

        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)

            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias, return_all_hidden_states=True
            )
            pocket_hidden_states = pocket_outputs[-1]
            layer_idx = getattr(model.args if hasattr(model, "args") else model, "atom_layer", -1)
            chosen_pocket_hidden = pocket_hidden_states[layer_idx]  # [B_p, seq_len, D]

            # global CLS for pocket
            pkt_cls = pocket_outputs[0][:, 0, :]
            pkt_global = model.pocket_project(pkt_cls)
            pkt_global = pkt_global / (pkt_global.norm(dim=-1, keepdim=True) + 1e-8)
            pkt_global_np = pkt_global.detach().cpu().numpy()
            pocket_global_list.extend([pkt_global_np[i] for i in range(pkt_global_np.shape[0])])

            # per-atom
            pocket_atom_tensor = chosen_pocket_hidden[:, 1:-1, :].detach()
            if pocket_padding_mask is not None:
                pocket_atom_mask = ~pocket_padding_mask[:, 1:-1]
            else:
                pocket_atom_mask = torch.ones(pocket_atom_tensor.size(0), pocket_atom_tensor.size(1), dtype=torch.bool, device=pocket_atom_tensor.device)

            Bp, P_i, D = pocket_atom_tensor.shape
            pocket_atom_flat = pocket_atom_tensor.reshape(-1, D)
            pocket_atom_proj_flat = model.pocket_project(pocket_atom_flat)
            Hp = pocket_atom_proj_flat.size(-1)
            pocket_atom_proj = pocket_atom_proj_flat.view(Bp, P_i, Hp)
            pocket_atom_proj = pocket_atom_proj / (pocket_atom_proj.norm(dim=-1, keepdim=True) + 1e-8)

            for i in range(Bp):
                valid_len = int(pocket_atom_mask[i].sum().item())
                if valid_len == 0:
                    pocket_atom_list.append(np.zeros((0, Hp), dtype=np.float32))
                else:
                    arr = pocket_atom_proj[i, :valid_len, :].detach().cpu().numpy()
                    pocket_atom_list.append(arr)

            pocket_names.extend(sample["pocket_name"])

        pocket_global_arr = np.stack(pocket_global_list, axis=0)  # [Np, Hg]

        # ---- topk-topk atom scoring (chunked and vectorized) ----
        import torch as _torch
        device = 'cuda' if _torch.cuda.is_available() else 'cpu'

        def compute_atom_scores_topk_chunked(pocket_list, mol_list, device=device, p_chunk=8, m_chunk=256, K=10):
            """
            Compute topk-topk score per pocket-molecule pair.
            pocket_list / mol_list: lists of numpy arrays [n_atoms, H]
            returns numpy array [Np, Nm]
            """
            Np = len(pocket_list)
            Nm = len(mol_list)
            if Np == 0 or Nm == 0:
                return np.zeros((Np, Nm), dtype=np.float32)
            H = pocket_list[0].shape[1]

            # precompute max lengths for chunk tensors
            Pmax = max(arr.shape[0] for arr in pocket_list) if pocket_list else 0
            Mmax = max(arr.shape[0] for arr in mol_list) if mol_list else 0

            # pad pocket and mol chunks and masks on device as needed in each chunk loop
            atom_scores = _torch.zeros((Np, Nm), dtype=_torch.float32, device=device)

            for p0 in range(0, Np, p_chunk):
                p1 = min(Np, p0 + p_chunk)
                # build pocket chunk padded
                pkt_chunk_list = pocket_list[p0:p1]
                Cp = p1 - p0
                Pchunk_max = max(arr.shape[0] for arr in pkt_chunk_list) if pkt_chunk_list else 0
                pkt_padded = _torch.zeros((Cp, Pchunk_max, H), dtype=_torch.float32, device=device)
                pkt_mask = _torch.zeros((Cp, Pchunk_max), dtype=_torch.bool, device=device)
                for i_idx, arr in enumerate(pkt_chunk_list):
                    L = arr.shape[0]
                    if L > 0:
                        pkt_padded[i_idx, :L, :] = _torch.from_numpy(arr).to(device)
                        pkt_mask[i_idx, :L] = 1

                for m0 in range(0, Nm, m_chunk):
                    m1 = min(Nm, m0 + m_chunk)
                    Cm = m1 - m0
                    mol_chunk_list = mol_list[m0:m1]
                    Mchunk_max = max(arr.shape[0] for arr in mol_chunk_list) if mol_chunk_list else 0
                    mol_padded = _torch.zeros((Cm, Mchunk_max, H), dtype=_torch.float32, device=device)
                    mol_mask = _torch.zeros((Cm, Mchunk_max), dtype=_torch.bool, device=device)
                    for j_idx, arr in enumerate(mol_chunk_list):
                        L = arr.shape[0]
                        if L > 0:
                            mol_padded[j_idx, :L, :] = _torch.from_numpy(arr).to(device)
                            mol_mask[j_idx, :L] = 1

                    # pairwise atom similarity: [Cp, Cm, Pchunk_max, Mchunk_max]
                    sim = _torch.einsum('pih,mjh->pmij', pkt_padded, mol_padded)  # float32 on device

                    # mask invalid positions
                    pkt_m = pkt_mask.view(pkt_mask.size(0), 1, pkt_mask.size(1), 1)
                    mol_m = mol_mask.view(1, mol_mask.size(0), 1, mol_mask.size(1))
                    valid_mask = pkt_m & mol_m
                    sim[~valid_mask] = -1e9

                    # topk over pocket atoms (dim=2) for each mol atom => vals_mol shape [Cp, Cm, k1, Mchunk_max]
                    k1 = min(K, sim.size(2))
                    if k1 > 0:
                        vals_mol, _ = _torch.topk(sim, k=k1, dim=2, largest=True, sorted=False)  # [Cp,Cm,k1,M]
                        # zero out sentinel padding contributions (we used -1e9 for invalid)
                        # any extremely negative entry likely corresponds to padding — treat it as 0 contribution
                        vals_mol = vals_mol.clone()
                        vals_mol[vals_mol < -1e8] = 0.0
                        # sum over the selected pocket-topk then mol dim -> [Cp, Cm]
                        sum_mol = vals_mol.sum(dim=2).sum(dim=2)
                    else:
                        sum_mol = _torch.zeros(sim.size(0), sim.size(1), device=device)

                    # topk over mol atoms (dim=3) for each pocket atom => vals_pkt shape [Cp, Cm, Pchunk_max, k2]
                    k2 = min(K, sim.size(3))
                    if k2 > 0:
                        vals_pkt, _ = _torch.topk(sim, k=k2, dim=3, largest=True, sorted=False)  # [Cp,Cm,P,k2]
                        vals_pkt = vals_pkt.clone()
                        vals_pkt[vals_pkt < -1e8] = 0.0
                        sum_pkt = vals_pkt.sum(dim=3).sum(dim=2)
                    else:
                        sum_pkt = _torch.zeros(sim.size(0), sim.size(1), device=device)

                    # denominator: for each pair (i,j) denom = min(K, valid_P_i) * valid_M_j + min(K, valid_M_j) * valid_P_i
                    valid_P = pkt_mask.sum(dim=1).to(_torch.float32)  # [Cp]
                    valid_M = mol_mask.sum(dim=1).to(_torch.float32)  # [Cm]
                    kp = _torch.clamp(valid_P.view(-1, 1), max=K)  # [Cp,1]
                    km = _torch.clamp(valid_M.view(1, -1), max=K)  # [1,Cm]
                    denom = kp * valid_M.view(1, -1) + km * valid_P.view(-1, 1)  # [Cp, Cm]
                    # avoid divide by zero
                    denom = denom.to(device)
                    total_sum = sum_mol + sum_pkt  # [Cp, Cm]
                    # compute normalized score; where denom==0 set score 0
                    denom_mask = denom > 0
                    score_chunk = _torch.zeros_like(total_sum)
                    score_chunk[denom_mask] = total_sum[denom_mask] / (denom[denom_mask] + 1e-12)

                    atom_scores[p0:p1, m0:m1] = score_chunk

                    # free memory
                    del sim, vals_mol, vals_pkt, sum_mol, sum_pkt, total_sum, denom, denom_mask, score_chunk

            return atom_scores.cpu().numpy()

        # get topk from args if provided, otherwise default 5
        #K = getattr(self.args, "topk_k", 5) if hasattr(self, "args") else 5
        atom_scores = compute_atom_scores_topk_chunked(pocket_atom_list, mol_atom_list, device=device, p_chunk=8, m_chunk=256, K=K)

        # --- combine with global and compute metrics ---
        global_scores = pocket_global_arr @ mol_global_arr.T  # [Np, Nm]
        # tuning knob: atom_interaction_weight; default conservative 0.1 if not set
        #alpha = getattr(self.args, "atom_interaction_weight", 0.1)
        # scale atom by model temperature if available (paper uses temperature for contrastive)
        # if hasattr(model, "logit_scale"):
        #     temp = float(model.logit_scale.exp().detach().cpu().item())
        #     atom_scores = atom_scores * temp
            
        # --- robust scaling + sanitization (protect against tiny std / NaNs / Infs) ---
        global_mean = np.mean(global_scores)
        global_std = np.std(global_scores)

        # compute atom std safely
        atom_mean = np.mean(atom_scores)
        atom_std = np.std(atom_scores)

        # small epsilon to avoid divide-by-zero
        eps = 1e-12

        # if atom_std is tiny, skip massive scaling (use scale=1.0)
        if not np.isfinite(atom_std) or atom_std < 1e-6:
            scale = 1.0
        else:
            scale = (global_std / (atom_std + eps))

        # cap scale to avoid huge multipliers (tunable)
        MAX_SCALE = 100.0
        if not np.isfinite(scale):
            scale = 1.0
        else:
            if abs(scale) > MAX_SCALE:
                scale = np.sign(scale) * MAX_SCALE

        atom_scores = atom_scores * scale

        # Replace NaN/Inf with finite numbers; choose a reasonable clipping window
        # Window based on global stats (±N sigma) keeps values comparable to global_scores
        CLIP_SIGMA = 10.0
        clip_min = global_mean - CLIP_SIGMA * (global_std + eps)
        clip_max = global_mean + CLIP_SIGMA * (global_std + eps)

        # Convert to finite and clip
        atom_scores = np.nan_to_num(atom_scores, nan=0.0, posinf=clip_max, neginf=clip_min)
        atom_scores = np.clip(atom_scores, clip_min, clip_max)

        # final sanity prints (optional, helpful for debugging)
        print("DEBUG global_scores mean/std:", np.mean(global_scores), np.std(global_scores))
        print("DEBUG atom_scores mean/std after scaling/clipping:", np.mean(atom_scores), np.std(atom_scores))
        print("DEBUG scale used:", scale)

        # combine
        combined = (1.0 - alpha) * global_scores + alpha * atom_scores

        res_single = combined.max(axis=0)
        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)
        print(f"alpha {alpha:.4f}  auc {auc:.6f}  bedroc {bedroc:.6f}")
        print("DEBUG global_scores mean/std:", np.mean(global_scores), np.std(global_scores))
        print("DEBUG atom_scores mean/std:", np.mean(atom_scores), np.std(atom_scores))
        print("DEBUG combined mean/std:", np.mean(combined), np.std(combined))
        print("DEBUG pocket_global shape, mol_global shape:", pocket_global_arr.shape, mol_global_arr.shape)
        print("DEBUG atom_scores shape:", atom_scores.shape)
        print("corr:", np.corrcoef(global_scores.flatten(), atom_scores.flatten())[0,1])
        print("example valid_P counts sample:", [arr.shape[0] for arr in pocket_atom_list[:10]])
        print("example valid_M counts sample:", [arr.shape[0] for arr in mol_atom_list[:10]])   
        #res_single = combined.max(axis=0)  # per-molecule best pocket score

        #auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)

        print(name)
        print(np.sum(labels), len(labels) - np.sum(labels))

        return auc, bedroc, ef_list, re_list, res_single, labels
    
    
    

    def test_pcba(self, model, **kwargs):
        #ckpt_date = self.args.finetune_from_model.split("/")[-2]
        #save_name = "/home/gaobowen/DrugClip/test_results/pcba/" + ckpt_date + ".txt"
        #save_name = ""
        
        targets = os.listdir("./data/lit_pcba/")
        K = getattr(self.args, "topk_k", 10) if hasattr(self, "args") else 10
        alphas = [0.01, 0.05, 0.1, 0.15]  # choose what you want

        #print(targets)
        for alpha in alphas:
            auc_list = []
            ef_list = []
            bedroc_list = []
            res_list= []
            labels_list = []
            re_list = {
                "0.005": [],
                "0.01": [],
                "0.02": [],
                "0.05": []
            }
            ef_list = {
                "0.005": [],
                "0.01": [],
                "0.02": [],
                "0.05": []
            }
            for i,target in enumerate(targets):
                auc, bedroc, ef, re, res_single, labels = self.test_pcba_target(target, model, alpha, K)
                auc_list.append(auc)
                bedroc_list.append(bedroc)
                for key in ef:
                    ef_list[key].append(ef[key])
                # print("re", re)
                # print("ef", ef)
                for key in re:
                    re_list[key].append(re[key])
                res_list.append(res_single)
                labels_list.append(labels)
            res = np.concatenate(res_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            print("Alpha for test_dude_target: ", alpha)
            # print(auc_list)
            # print(ef_list)
            # print("auc 25%", np.percentile(auc_list, 25))
            # print("auc 50%", np.percentile(auc_list, 50))
            # print("auc 75%", np.percentile(auc_list, 75))
            print("auc mean", np.mean(auc_list))
            # print("bedroc 25%", np.percentile(bedroc_list, 25))
            # print("bedroc 50%", np.percentile(bedroc_list, 50))
            # print("bedroc 75%", np.percentile(bedroc_list, 75))
            print("bedroc mean", np.mean(bedroc_list))
            #print(np.median(auc_list))
            #print(np.median(ef_list))
            for key in ef_list:
                # print("ef", key, "25%", np.percentile(ef_list[key], 25))
                # print("ef",key, "50%", np.percentile(ef_list[key], 50))
                # print("ef",key, "75%", np.percentile(ef_list[key], 75))
                print("ef",key, "mean", np.mean(ef_list[key]))
            for key in re_list:
                # print("re",key, "25%", np.percentile(re_list[key], 25))
                # print("re",key, "50%", np.percentile(re_list[key], 50))
                # print("re",key, "75%", np.percentile(re_list[key], 75))
                print("re",key, "mean", np.mean(re_list[key]))

        return  
    
    def test_dude_target(self, target, model, alpha=0.0, K=10, **kwargs):
        data_path = "./data/DUD-E/raw/all/" + target + "/mols.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz = 64
        print(num_data // bsz)

        # Lists to collect
        mol_atom_list = []      # list of numpy arrays [n_atoms, H]
        mol_global_list = []    # list of numpy arrays [H_global] (global proj)
        mol_names = []
        labels = []

        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)

            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias, return_all_hidden_states=True
            )
            mol_cls = mol_outputs[0][:, 0, :]                    # [B, D]
            mol_global = model.mol_project(mol_cls)              # [B, Hg]
            mol_global = mol_global / (mol_global.norm(dim=-1, keepdim=True )+ 1e-8)
            mol_global_np = mol_global.detach().cpu().numpy()
            mol_global_list.extend([mol_global_np[i] for i in range(mol_global_np.shape[0])])
            hidden_states = mol_outputs[-1]
            #print("num_hidden_states:", len(hidden_states))
            #for i, h in enumerate(hidden_states[-4:], start=len(hidden_states)-4):  # last 4
                #print("layer idx", i, "shape", h.shape)
            # quick peek of mean activations
            #print("last layer mean:", hidden_states[-1].mean().item())
            #print("second-last mean:", hidden_states[-2].mean().item())

            layer_idx = getattr(model.args if hasattr(model, "args") else model, "atom_layer", -1)
            chosen_hidden = hidden_states[layer_idx]  # [B, seq_len, D]

            # --- PER-ATOM embeddings (exclude BOS/EOS tokens at positions 0 and -1) ---
            mol_atom_tensor = chosen_hidden[:, 1:-1, :].detach()  # [B, M_i, D]
            if mol_padding_mask is not None:
                mol_atom_mask = ~mol_padding_mask[:, 1:-1]
            else:
                mol_atom_mask = torch.ones(mol_atom_tensor.size(0), mol_atom_tensor.size(1), dtype=torch.bool, device=mol_atom_tensor.device)

            B, M_i, D = mol_atom_tensor.shape
            mol_atom_flat = mol_atom_tensor.reshape(-1, D)  # [B*M_i, D]
            mol_atom_proj_flat = model.mol_project(mol_atom_flat)  # [B*M_i, H]
            H = mol_atom_proj_flat.size(-1)
            mol_atom_proj = mol_atom_proj_flat.view(B, M_i, H)  # [B, M_i, H]
            mol_atom_proj = mol_atom_proj / (mol_atom_proj.norm(dim=-1, keepdim=True) + 1e-8)

            # Save each molecule's per-atom vectors (only valid atoms)
            for i in range(B):
                valid_len = int(mol_atom_mask[i].sum().item())
                if valid_len == 0:
                    mol_atom_list.append(np.zeros((0, H), dtype=np.float32))
                else:
                    arr = mol_atom_proj[i, :valid_len, :].detach().cpu().numpy()
                    mol_atom_list.append(arr)

            mol_names.extend(sample["smi_name"])
            labels.extend(sample["target"].detach().cpu().numpy())

        # Convert labels and global embeddings
        mol_global_arr = np.stack(mol_global_list, axis=0)  # [Nm, Hg]
        labels = np.array(labels, dtype=np.int32)

        # --- POCKETS ---
        data_path = "./data/DUD-E/raw/all/" + target + "/pocket.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)

        pocket_atom_list = []
        pocket_global_list = []
        pocket_names = []

        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)

            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias, return_all_hidden_states=True
            )
            pocket_hidden_states = pocket_outputs[-1]
            layer_idx = getattr(model.args if hasattr(model, "args") else model, "atom_layer", -1)
            chosen_pocket_hidden = pocket_hidden_states[layer_idx]  # [B_p, seq_len, D]
            # global CLS for pocket
            pkt_cls = pocket_outputs[0][:, 0, :]                 # final x's CLS token
            pkt_global = model.pocket_project(pkt_cls)
            pkt_global = pkt_global / (pkt_global.norm(dim=-1, keepdim=True)+ 1e-8)
            pkt_global_np = pkt_global.detach().cpu().numpy()
            pocket_global_list.extend([pkt_global_np[i] for i in range(pkt_global_np.shape[0])])
            
            # per-atom
            pocket_atom_tensor = chosen_pocket_hidden[:, 1:-1, :].detach()
            if pocket_padding_mask is not None:
                pocket_atom_mask = ~pocket_padding_mask[:, 1:-1]
            else:
                pocket_atom_mask = torch.ones(pocket_atom_tensor.size(0), pocket_atom_tensor.size(1), dtype=torch.bool, device=pocket_atom_tensor.device)

            Bp, P_i, D = pocket_atom_tensor.shape
            pocket_atom_flat = pocket_atom_tensor.reshape(-1, D)
            pocket_atom_proj_flat = model.pocket_project(pocket_atom_flat)
            Hp = pocket_atom_proj_flat.size(-1)
            pocket_atom_proj = pocket_atom_proj_flat.view(Bp, P_i, Hp)
            pocket_atom_proj = pocket_atom_proj / (pocket_atom_proj.norm(dim=-1, keepdim=True) + 1e-8)
                
            for i in range(Bp):
                valid_len = int(pocket_atom_mask[i].sum().item())
                if valid_len == 0:
                    pocket_atom_list.append(np.zeros((0, Hp), dtype=np.float32))
                else:
                    arr = pocket_atom_proj[i, :valid_len, :].detach().cpu().numpy()
                    pocket_atom_list.append(arr)
            pocket_names.extend(sample["pocket_name"])

        pocket_global_arr = np.stack(pocket_global_list, axis=0)  # [Np, Hg]

        # ---- topk-topk atom scoring (chunked and vectorized) ----
        import torch as _torch
        device = 'cuda' if _torch.cuda.is_available() else 'cpu'

        def compute_atom_scores_topk_chunked(pocket_list, mol_list, device=device, p_chunk=8, m_chunk=256, K=10):
            """
            Compute topk-topk score per pocket-molecule pair.
            pocket_list / mol_list: lists of numpy arrays [n_atoms, H]
            returns numpy array [Np, Nm]
            """
            Np = len(pocket_list)
            Nm = len(mol_list)
            if Np == 0 or Nm == 0:
                return np.zeros((Np, Nm), dtype=np.float32)
            H = pocket_list[0].shape[1]

            # precompute max lengths for chunk tensors
            Pmax = max(arr.shape[0] for arr in pocket_list) if pocket_list else 0
            Mmax = max(arr.shape[0] for arr in mol_list) if mol_list else 0

            # pad pocket and mol chunks and masks on device as needed in each chunk loop
            atom_scores = _torch.zeros((Np, Nm), dtype=_torch.float32, device=device)

            for p0 in range(0, Np, p_chunk):
                p1 = min(Np, p0 + p_chunk)
                # build pocket chunk padded
                pkt_chunk_list = pocket_list[p0:p1]
                Cp = p1 - p0
                Pchunk_max = max(arr.shape[0] for arr in pkt_chunk_list) if pkt_chunk_list else 0
                pkt_padded = _torch.zeros((Cp, Pchunk_max, H), dtype=_torch.float32, device=device)
                pkt_mask = _torch.zeros((Cp, Pchunk_max), dtype=_torch.bool, device=device)
                for i_idx, arr in enumerate(pkt_chunk_list):
                    L = arr.shape[0]
                    if L > 0:
                        pkt_padded[i_idx, :L, :] = _torch.from_numpy(arr).to(device)
                        pkt_mask[i_idx, :L] = 1

                for m0 in range(0, Nm, m_chunk):
                    m1 = min(Nm, m0 + m_chunk)
                    Cm = m1 - m0
                    mol_chunk_list = mol_list[m0:m1]
                    Mchunk_max = max(arr.shape[0] for arr in mol_chunk_list) if mol_chunk_list else 0
                    mol_padded = _torch.zeros((Cm, Mchunk_max, H), dtype=_torch.float32, device=device)
                    mol_mask = _torch.zeros((Cm, Mchunk_max), dtype=_torch.bool, device=device)
                    for j_idx, arr in enumerate(mol_chunk_list):
                        L = arr.shape[0]
                        if L > 0:
                            mol_padded[j_idx, :L, :] = _torch.from_numpy(arr).to(device)
                            mol_mask[j_idx, :L] = 1

                    # pairwise atom similarity: [Cp, Cm, Pchunk_max, Mchunk_max]
                    sim = _torch.einsum('pih,mjh->pmij', pkt_padded, mol_padded)  # float32 on device

                    # mask invalid positions
                    pkt_m = pkt_mask.view(pkt_mask.size(0), 1, pkt_mask.size(1), 1)
                    mol_m = mol_mask.view(1, mol_mask.size(0), 1, mol_mask.size(1))
                    valid_mask = pkt_m & mol_m
                    sim[~valid_mask] = -1e9

                    # topk over pocket atoms (dim=2) for each mol atom => vals_mol shape [Cp, Cm, k1, Mchunk_max]
                    k1 = min(K, sim.size(2))
                    if k1 > 0:
                        vals_mol, _ = _torch.topk(sim, k=k1, dim=2, largest=True, sorted=False)  # [Cp,Cm,k1,M]
                        # zero out sentinel padding contributions (we used -1e9 for invalid)
                        # any extremely negative entry likely corresponds to padding — treat it as 0 contribution
                        vals_mol = vals_mol.clone()
                        vals_mol[vals_mol < -1e8] = 0.0
                        # sum over the selected pocket-topk then mol dim -> [Cp, Cm]
                        sum_mol = vals_mol.sum(dim=2).sum(dim=2)
                    else:
                        sum_mol = _torch.zeros(sim.size(0), sim.size(1), device=device)

                    # topk over mol atoms (dim=3) for each pocket atom => vals_pkt shape [Cp, Cm, Pchunk_max, k2]
                    k2 = min(K, sim.size(3))
                    if k2 > 0:
                        vals_pkt, _ = _torch.topk(sim, k=k2, dim=3, largest=True, sorted=False)  # [Cp,Cm,P,k2]
                        vals_pkt = vals_pkt.clone()
                        vals_pkt[vals_pkt < -1e8] = 0.0
                        sum_pkt = vals_pkt.sum(dim=3).sum(dim=2)
                    else:
                        sum_pkt = _torch.zeros(sim.size(0), sim.size(1), device=device)

                    # denominator: for each pair (i,j) denom = min(K, valid_P_i) * valid_M_j + min(K, valid_M_j) * valid_P_i
                    valid_P = pkt_mask.sum(dim=1).to(_torch.float32)  # [Cp]
                    valid_M = mol_mask.sum(dim=1).to(_torch.float32)  # [Cm]
                    kp = _torch.clamp(valid_P.view(-1, 1), max=K)  # [Cp,1]
                    km = _torch.clamp(valid_M.view(1, -1), max=K)  # [1,Cm]
                    denom = kp * valid_M.view(1, -1) + km * valid_P.view(-1, 1)  # [Cp, Cm]
                    # avoid divide by zero
                    denom = denom.to(device)
                    total_sum = sum_mol + sum_pkt  # [Cp, Cm]
                    # compute normalized score; where denom==0 set score 0
                    denom_mask = denom > 0
                    score_chunk = _torch.zeros_like(total_sum)
                    score_chunk[denom_mask] = total_sum[denom_mask] / (denom[denom_mask] + 1e-12)

                    atom_scores[p0:p1, m0:m1] = score_chunk

                    # free memory
                    del sim, vals_mol, vals_pkt, sum_mol, sum_pkt, total_sum, denom, denom_mask, score_chunk

            return atom_scores.cpu().numpy()

        # get topk from args if provided, otherwise default 5
        #K = getattr(self.args, "topk_k", 5) if hasattr(self, "args") else 5
        atom_scores = compute_atom_scores_topk_chunked(pocket_atom_list, mol_atom_list, device=device, p_chunk=8, m_chunk=256, K=K)

        # --- combine with global and compute metrics ---
        global_scores = pocket_global_arr @ mol_global_arr.T  # [Np, Nm]
        # tuning knob: atom_interaction_weight; default conservative 0.1 if not set
        #alpha = getattr(self.args, "atom_interaction_weight", 0.1)
        # scale atom by model temperature if available (paper uses temperature for contrastive)
        # if hasattr(model, "logit_scale"):
        #     temp = float(model.logit_scale.exp().detach().cpu().item())
        #     atom_scores = atom_scores * temp
            
        global_std = np.std(global_scores)
        atom_std = np.std(atom_scores) + 1e-12
        scale = (global_std / atom_std)
        atom_scores = atom_scores * scale

        combined = (1.0 - alpha) * global_scores + alpha * atom_scores
        res_single = combined.max(axis=0)
        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)
        print(f"alpha {alpha:.4f}  auc {auc:.6f}  bedroc {bedroc:.6f}")
        print("DEBUG global_scores mean/std:", np.mean(global_scores), np.std(global_scores))
        print("DEBUG atom_scores mean/std:", np.mean(atom_scores), np.std(atom_scores))
        print("DEBUG combined mean/std:", np.mean(combined), np.std(combined))
        print("DEBUG pocket_global shape, mol_global shape:", pocket_global_arr.shape, mol_global_arr.shape)
        print("DEBUG atom_scores shape:", atom_scores.shape)
        print("corr:", np.corrcoef(global_scores.flatten(), atom_scores.flatten())[0,1])
        print("example valid_P counts sample:", [arr.shape[0] for arr in pocket_atom_list[:10]])
        print("example valid_M counts sample:", [arr.shape[0] for arr in mol_atom_list[:10]])   
        #res_single = combined.max(axis=0)  # per-molecule best pocket score

        #auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)

        print(target)
        print(np.sum(labels), len(labels) - np.sum(labels))

        return auc, bedroc, ef_list, re_list, res_single, labels



    def test_dude(self, model, **kwargs):


        targets = os.listdir("./data/DUD-E/raw/all/")
        K = getattr(self.args, "topk_k", 10) if hasattr(self, "args") else 10
        alphas = [0.01, 0.05, 0.1, 0.15]  # choose what you want
        for alpha in alphas:
            auc_list = []
            bedroc_list = []
            ef_list = []
            res_list= []
            labels_list = []
            re_list = {
                "0.005": [],
                "0.01": [],
                "0.02": [],
                "0.05": [],
            }
            ef_list = {
                "0.005": [],
                "0.01": [],
                "0.02": [],
                "0.05": [],
            }
            for i,target in enumerate(targets):
                auc, bedroc, ef, re, res_single, labels = self.test_dude_target(target, model, alpha, K)
                auc_list.append(auc)
                bedroc_list.append(bedroc)
                for key in ef:
                    ef_list[key].append(ef[key])
                for key in re_list:
                    re_list[key].append(re[key])
                res_list.append(res_single)
                labels_list.append(labels)
            res = np.concatenate(res_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            print("Alpha for test_dude_target: ", alpha)
            print("auc mean", np.mean(auc_list))
            print("bedroc mean", np.mean(bedroc_list))

            for key in ef_list:
                print("ef", key, "mean", np.mean(ef_list[key]))

            for key in re_list:
                print("re", key, "mean",  np.mean(re_list[key]))

            # save printed results 
        
        
        return
    
    
    def encode_mols_once(self, model, data_path, emb_dir, atoms, coords, **kwargs):
        
        # cache path is embdir/data_path.pkl

        #cache_path = os.path.join(emb_dir, data_path.split("/")[-1] + ".pkl")

        #if os.path.exists(cache_path):
            #with open(cache_path, "rb") as f:
                #mol_reps, mol_names = pickle.load(f)
            #return mol_reps, mol_names

        mol_dataset = self.load_retrieval_mols_dataset(data_path,atoms,coords)
        mol_reps = []
        mol_names = []
        bsz=32
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
            mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias, return_all_hidden_states=True
            )
            hidden_states = mol_outputs[-1]
            layer_idx = getattr(model.args if hasattr(model, "args") else model, "atom_layer", -1)
            chosen_hidden = hidden_states[layer_idx]
            print("mol_outputs in tasks/drugclip.py encode_mols_once DEBUG chosen_hidden shape:", chosen_hidden.shape)  # expect [B, seq_len, D] for per-atom
            print("DEBUG pooled shape (CLS):", chosen_hidden[:,0,:].shape)
            mol_encoder_rep = chosen_hidden[:, 0, :]
            mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])

        mol_reps = np.concatenate(mol_reps, axis=0)

        # save the results
        
        #with open(cache_path, "wb") as f:
            #pickle.dump([mol_reps, mol_names], f)

        return mol_reps, mol_names
    
    def retrieve_mols(self, model, mol_path, pocket_path, emb_dir, k, **kwargs):
 
        #os.makedirs(emb_dir, exist_ok=True)        
        mol_reps, mol_names = self.encode_mols_once(model, mol_path, emb_dir,  "atoms", "coordinates")
        
        pocket_dataset = self.load_pockets_dataset(pocket_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=16, collate_fn=pocket_dataset.collater)
        pocket_reps = []
        pocket_names = []
        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
            pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias, return_all_hidden_states=True
            )
            pocket_hidden_states = pocket_outputs[-1]
            layer_idx = getattr(model.args if hasattr(model, "args") else model, "atom_layer", -1)
            chosen_pocket_hidden = pocket_hidden_states[layer_idx]
            pocket_encoder_rep = chosen_pocket_hidden[:, 0, :]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
            pocket_names.extend(sample["pocket_name"])
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        
        res = pocket_reps @ mol_reps.T
        res = res.max(axis=0)


        # get top k results

        
        top_k = np.argsort(res)[::-1][:k]

        # return names and scores
        
        return [mol_names[i] for i in top_k], res[top_k]