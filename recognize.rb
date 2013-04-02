require 'mlp'
require 'pry'
require 'chunky_png'

require_relative 'noise'

Noise.apply 20

mlp = MLP.new(:hidden_layers => [3], :output_nodes => 3, :inputs => 100)

images = {}
Dir['patterns/*.png'].each do |file|
  png = ChunkyPNG::Image.from_file(file)
  images[file] = png.pixels.map { |p| p < 1000 ? 0 : 1 }
end

def expected(file)
  [(file =~ /K/ ? 1 : 0),(file =~ /Y/ ? 1 : 0), (file =~ /P/ ? 1 : 0)]
end

def get_letter(result)
  case result.max
  when result[0]
    'K'
  when result[1]
    'Y'
  when result[2]
    'P'
  end
end

1000.times do |i|
  errors = images.map do |file, pixels|
    mlp.train pixels.first(100), expected(file)
  end

  puts "Error after iteration #{i}:\t#{errors.max}" if i % 100 == 0
end

Dir['recognize/*.png'].each_with_index do |file, i|
  png = ChunkyPNG::Image.from_file(file)
  pixels = png.pixels.map { |p| p < 1000 ? 0 : 1 }

  result = mlp.feed_forward(pixels)

  png.save file.gsub(/\/.*\.png/, "/#{get_letter result}_#{'%5f' % result.max}_#{i}.png")
  File.delete file
end
