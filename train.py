from model import CustomSchedule, TransformerContext, create_masks
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import time
import os


def encode_source(lang1):
  lang1 = [tokenizer_src.vocab_size] + tokenizer_src.encode(
      lang1.numpy()) + [tokenizer_src.vocab_size+1]
  return [lang1]

def encode_target(lang1):
  lang1 = [tokenizer_tgt.vocab_size] + tokenizer_tgt.encode(
      lang1.numpy()) + [tokenizer_tgt.vocab_size+1]
  return [lang1]

def new_encode_source(line):
  (encoded_text)= tf.py_function(encode_source, inp= [line], Tout=(tf.int64))
  return encoded_text

def new_encode_target(line):
  return tf.py_function(encode_target, inp= [line], Tout=(tf.int64))

def filter_max_length(x, y,z, max_length=70):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_mean(loss_)

@tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            ])
def train_step(inp, tar,con):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask,con_padding_mask  = create_masks(inp, tar_inp,con)
  
  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, con,
                                 True, 
                                 enc_padding_mask,
                                  con_padding_mask,
                                 combined_mask, 
                                 dec_padding_mask,activate_context)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  train_accuracy(tar_real, predictions)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create SubwordTextEncoder\
                                     for languages')
    parser.add_argument('-p', '--path', help='Path where traing sentences are\
                      (src,tgt,context)')
    parser.add_argument('-s', '--src', help='Source language code')
    parser.add_argument('-t', '--tgt', help='Target language code')
    parser.add_argument('-checkpoint', '--checkpoint', help='Checkpoint path')
    parser.add_argument('-c', '--context', help='Activate context-aware\
                         mechanisms', action="store_true")
    parser.add_argument('-l', '--layers', help='Number of stacks of layers in\
                         encoder/decoder', default=3, type=int)
    parser.add_argument('-heads', '--heads', help='Number of attention heads',\
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
    num_layers = vars['layers']
    d_model = vars['dimension']
    dff = vars['inner_dim']
    num_heads = vars['heads']
    dropout_rate = vars['dropout']
    activate_context = vars['context']
    checkpoint_path = vars['checkpoint']

    tokenizer_src = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
                                                f"{path}/vocab_encoder_{src}")
    tokenizer_tgt = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
                                                f"{path}/vocab_encoder_{tgt}")
    input_vocab_size = tokenizer_src.vocab_size + 2
    target_vocab_size = tokenizer_tgt.vocab_size + 2

    BUFFER_SIZE = 20000
    MAX_LENGTH = 70

    if activate_context:
        BATCH_SIZE = 64
        EPOCHS = 5
    else:
        BATCH_SIZE = 200
        EPOCHS = 2

    train_data_src = tf.data.TextLineDataset(f'{path}/train.{src}')
    train_data_tgt = tf.data.TextLineDataset(f'{path}/train.{tgt}')
    train_data_context = tf.data.TextLineDataset(f'{path}/train.context')

    AUTOTUNE=tf.data.experimental.AUTOTUNE
    dataset_src = train_data_src.map(new_encode_source, \
                                        num_parallel_calls=AUTOTUNE)
    dataset_context = train_data_context.map(new_encode_source, \
                                        num_parallel_calls=AUTOTUNE)
    dataset_tgt = train_data_tgt.map(new_encode_target)

    train_dataset = tf.data.Dataset.zip((dataset_src,dataset_tgt, \
                                        dataset_context))
    train_dataset = train_dataset.filter(filter_max_length)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.padded_batch(BATCH_SIZE, \
                                        padded_shapes=([-1], [-1],[-1]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,\
                                        beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                                from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                                            name='train_accuracy')

    transformer = TransformerContext(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate, activate_context=activate_context)

    ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    
    #2step-training
    if activate_context:
        for layer in transformer.layers:
            print(layer.name)
            if not (layer.name == "encoder_1" or layer.name == "encoder_context"):
                layer.trainable = False
    
    for epoch in range(EPOCHS):
        start = time.time()
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        # inp -> esp, tar -> english
        for (batch, (inp, tar, con)) in enumerate(train_dataset):
            train_step(inp, tar, con)
            
            if batch % 200 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                ckpt_save_path))
            
        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                        train_loss.result(), 
                                                        train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))



    
                                       
