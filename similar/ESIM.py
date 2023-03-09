def get_model(embedding_matrix_file, MAX_SEQUENCE_LENGTH, num_lstm, rate_drop_dense, num_dense):
 
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
 
    # embedding
    embedding_layer = create_pretrained_embedding(embedding_matrix_file, mask_zero=False)
    bn = BatchNormalization(axis=2)
    embedded_sequences_1 = bn(embedding_layer(sequence_1_input))
    embedded_sequences_2 = bn(embedding_layer(sequence_2_input))
 
    # encode
    encode = Bidirectional(LSTM(num_lstm, return_sequences=True))
    encode_sequences_1 = encode(embedded_sequences_1)
    encode_sequences_2 = encode(embedded_sequences_2)
 
    # attention
    alignd_sequences_1, alignd_sequences_2 = soft_attention_alignment(encode_sequences_1, encode_sequences_2)
 
    # compose
    combined_sequences_1 = Concatenate()(
        [encode_sequences_1, alignd_sequences_2, submult(encode_sequences_1, alignd_sequences_2)])
    combined_sequences_2 = Concatenate()(
        [encode_sequences_2, alignd_sequences_1, submult(encode_sequences_2, alignd_sequences_1)])
 
    compose = Bidirectional(LSTM(num_lstm, return_sequences=True))
    compare_sequences_1 = compose(combined_sequences_1)
    compare_sequences_2 = compose(combined_sequences_2)
 
    # aggregate
    rep_sequences_1 = apply_multiple(compare_sequences_1, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    rep_sequences_2 = apply_multiple(compare_sequences_2, [GlobalAvgPool1D(), GlobalMaxPool1D()])
 
    # classifier
    merged = Concatenate()([rep_sequences_1, rep_sequences_2])
    dense = BatchNormalization()(merged)
    dense = Dense(num_dense, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(rate_drop_dense)(dense)
    dense = Dense(num_dense, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(rate_drop_dense)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)
 
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=out_)
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['binary_crossentropy', 'accuracy'])
    return model