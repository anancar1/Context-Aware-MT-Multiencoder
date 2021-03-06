import argparse
import time
from bleu import list_bleu
from model import TransformerContext, create_masks, CustomSchedule
import tensorflow as tf
import tensorflow_datasets as tfds

def evaluate(inp_sentence,context_sentence,activate_context):
  start_token = [tokenizer_src.vocab_size]
  end_token = [tokenizer_src.vocab_size + 1]
  

  inp_sentence = start_token + tokenizer_src.encode(inp_sentence) + end_token
  context_sentence = start_token + tokenizer_src.encode(context_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)
  context_input = tf.expand_dims(context_sentence, 0)
  
  decoder_input = [tokenizer_tgt.vocab_size]
  output = tf.expand_dims(decoder_input, 0)
    
  for i in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask,con_padding_mask = create_masks(
        encoder_input, output,context_input)
  
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input, 
                                                 output,context_input,
                                                 False,
                                                 enc_padding_mask,con_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask,activate_context)
    
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    if predicted_id == tokenizer_tgt.vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights

def translate(sentence,context,activate_context):
    result, attention_weights = evaluate(sentence,context,activate_context)
  
    predicted_sentence = tokenizer_tgt.decode([i for i in result 
                                            if i < tokenizer_tgt.vocab_size])  

    return predicted_sentence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create SubwordTextEncoder\
                                     for languages')
    parser.add_argument('-p', '--path', help='Path where test sentences are\
                      (src,tgt,context)')
    parser.add_argument('-s', '--src', help='Source language code')
    parser.add_argument('-t', '--tgt', help='Target language code')
    parser.add_argument('-checkpoint', '--checkpoint', help='Path where checkpoints are stored')
    parser.add_argument('-c', '--context', help='Activate context-aware\
                         mechanisms', action="store_true")
    parser.add_argument('-l', '--layers', help='Number of stacks of layers in\
                         encoder/decoder', default=3, type=int)
    parser.add_argument('-he', '--heads', help='Number of attention heads',\
                        default=8, type=int)
    parser.add_argument('-d_model', '--dimension', help='Dim of the outputs\
                        (d_model)', default=512, type=int)
    parser.add_argument('-dff', '--inner_dim', help='Dim of inner-layer (dff)', \
                        default=2048, type=int)
    parser.add_argument('-drop', '--dropout', help='Dropout rate',\
                        default=0.3, type=int)
    args = parser.parse_args()
    vars = vars(args)
    path = vars['path']
    src = vars['src']
    tgt = vars['tgt']
    checkpoint_path = vars['checkpoint']
    num_layers = vars['layers']
    d_model = vars['dimension']
    dff = vars['inner_dim']
    num_heads = vars['heads']
    dropout_rate = vars['dropout']
    activate_context = vars['context']
    
    MAX_LENGTH = 70

    tokenizer_src = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
                                                f"{path}/vocab_encoder_{src}")
    tokenizer_tgt = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
                                                f"{path}/vocab_encoder_{tgt}")
    input_vocab_size = tokenizer_src.vocab_size + 2
    target_vocab_size = tokenizer_tgt.vocab_size + 2

    test_tgt = []
    with open(f'{path}/test.{tgt}') as f:
        lines = f.read().split("\n")[:-1]
        for line in lines: 
            test_tgt.append(line)

    test_src = []
    with open(f'{path}/test.{src}') as f:
        lines = f.read().split("\n")[:-1]
        for line in lines: 
            test_src.append(line)

    test_context = []
    with open(f'{path}/test.context') as f:
        lines = f.read().split("\n")[:-1]
        for line in lines: 
            test_context.append(line)

    transformer = TransformerContext(num_layers, d_model, num_heads, dff,
                    input_vocab_size, target_vocab_size, 
                    pe_input=input_vocab_size, 
                    pe_target=target_vocab_size,
                    rate=dropout_rate, activate_context=activate_context)
    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
    ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    if activate_context:
      ckpt.restore(ckpt_manager.latest_checkpoint)
    else:
      ckpt.restore(ckpt_manager.checkpoints[-2])


    hyp_en= []
    for i in range(len(test_src)):
        start = time.time()
        input_sentence = test_src[i]
        context_sentence = test_context[i]
        translated = translate(input_sentence, context_sentence,activate_context)
        hyp_en.append(translated)
        print(f'{i + 1}/{len(test_src)} - time taken for 1 translation:{"{:.2f}".format(time.time() - start)}')
    print("bleu:")
    print(list_bleu([test_tgt],hyp_en))

