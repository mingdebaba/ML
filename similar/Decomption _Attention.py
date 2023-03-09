def get_model(embedding_matrix_file, MAX_SEQUENCE_LENGTH,
              rate_drop_projction, num_projction, hidden_projction,
              rate_drop_compare, num_compare,
              rate_drop_dense, num_dense):
 
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
 
    # embedding
    embedding_layer = create_pretrained_embedding(embedding_matrix_file, mask_zero=False)
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    embedded_sequences_2 = embedding_layer(sequence_2_input)
 
    # projection
    projection_layers = []
    if hidden_projction > 0:
        projection_layers.extend([
            Dense(hidden_projction, activation='elu'),
            Dropout(rate=rate_drop_projction),
        ])
    projection_layers.extend([
        Dense(num_projction, activation=None),
        Dropout(rate=rate_drop_projction),
    ])
    encode_sequences_1 = time_distributed(embedded_sequences_1, projection_layers)
    encode_sequences_2 = time_distributed(embedded_sequences_2, projection_layers)
 
    # attention
    alignd_sequences_1, alignd_sequences_2 = soft_attention_alignment(encode_sequences_1, encode_sequences_2)
 
    # compare
    combined_sequences_1 = Concatenate()(
        [encode_sequences_1, alignd_sequences_2, submult(encode_sequences_1, alignd_sequences_2)])
    combined_sequences_2 = Concatenate()(
        [encode_sequences_2, alignd_sequences_1, submult(encode_sequences_2, alignd_sequences_1)])
    compare_layers = [
        Dense(num_compare, activation='elu'),
        Dropout(rate_drop_compare),
        Dense(num_compare, activation='elu'),
        Dropout(rate_drop_compare),
    ]
    compare_sequences_1 = time_distributed(combined_sequences_1, compare_layers)
    compare_sequences_2 = time_distributed(combined_sequences_2, compare_layers)
 
    # aggregate
    rep_sequences_1 = apply_multiple(compare_sequences_1, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    rep_sequences_2 = apply_multiple(compare_sequences_2, [GlobalAvgPool1D(), GlobalMaxPool1D()])
 
    # classifier
    merged = Concatenate()([rep_sequences_1, rep_sequences_2])
    dense = BatchNormalization()(merged)
    dense = Dense(num_dense, activation='elu')(dense)
    dense = Dropout(rate_drop_dense)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(num_dense, activation='elu')(dense)
    dense = Dropout(rate_drop_dense)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)
 
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=out_)
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['binary_crossentropy', 'accuracy'])
    return model