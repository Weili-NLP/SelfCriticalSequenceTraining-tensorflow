from core.rl_solve_debug import CaptioningSolver
from core.rl_model import CaptionGenerator
from core.utils import load_coco_data


def main():
    # load train dataset
    data = load_coco_data(data_path='./data', split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path='./data', split='val')

    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=256,
                                       dim_hidden=256, n_time_step=16, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, val_data, n_epochs=50000, batch_size=50, update_rule='adam',
                                          learning_rate=0.00005, print_every=1000, save_every=1, image_path='./image/',
                                    pretrained_model=None, model_path='./model/lstm/', test_model='/ais/gobi5/linghuan/basic-attention/model/lstm/lstm/model-19',
                                     print_bleu=True, log_path='./log/')

    solver.train()

if __name__ == "__main__":
    main()