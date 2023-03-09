#LSTM实现文本相似度：
def get_model(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH,
                           num_lstm, rate_drop_lstm, rate_drop_dense, num_dense, act):
 
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	
	# embedding
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    embedded_sequences_2 = embedding_layer(sequence_2_input)
	
	# lstm
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
    x1 = lstm_layer(embedded_sequences_1)
    y1 = lstm_layer(embedded_sequences_2)
 
	# classifier
    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)
 
    model = Model(inputs=[sequence_1_input, sequence_2_input], \
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    return model