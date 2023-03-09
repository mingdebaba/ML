#6. 使用多头自注意力机制的简单网络实现文本相似度
def get_model(embedding_matrix_file, MAX_SEQUENCE_LENGTH, rate_drop_dense):
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
 
    # embedding
    embedding_layer = create_pretrained_embedding(embedding_matrix_file, mask_zero=False)
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    embedded_sequences_2 = embedding_layer(sequence_2_input)
 
    # position embedding
    # embedded_sequences_1 = pos_embed.Position_Embedding()(embedded_sequences_1)
    # embedded_sequences_2 = pos_embed.Position_Embedding()(embedded_sequences_2)
 
 
    # attention
    O_seq_1 = Attention.Attention(8, 16)([embedded_sequences_1, embedded_sequences_1, embedded_sequences_1])
    O_seq_2 = Attention.Attention(8, 16)([embedded_sequences_2, embedded_sequences_2, embedded_sequences_2])
 
    # aggregate  ESMI
    # rep_sequences_1 = apply_multiple(compare_sequences_1, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    # rep_sequences_2 = apply_multiple(compare_sequences_2, [GlobalAvgPool1D(), GlobalMaxPool1D()])
 
    rep_sequences_1 = GlobalAveragePooling1D()(O_seq_1)
    rep_sequences_2 = GlobalAveragePooling1D()(O_seq_2)
 
    # classifier
    merged = Concatenate()([rep_sequences_1, rep_sequences_2])
    O_seq = Dropout(rate_drop_dense)(merged)
    outputs = Dense(1, activation='sigmoid')(O_seq)
 
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model