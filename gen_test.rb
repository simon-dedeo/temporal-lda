require "fileutils"
srand(42)
V = 18
words = %w[a b c d e f g h i A B C D E F G H I]

# Gamma(alpha,1) via Marsaglia & Tsang (alpha>=1) + Ahrens-Dieter shift (alpha<1)
def rgamma(alpha)
  if alpha < 1.0
    # Ahrens-Dieter: Gamma(a,1) = Gamma(a+1,1) * U^(1/a)
    return rgamma(alpha + 1.0) * (rand ** (1.0 / alpha))
  end
  d = alpha - 1.0/3.0
  c = 1.0 / Math.sqrt(9.0 * d)
  loop do
    x = nil; v = nil
    loop do
      x = rand_normal
      v = (1.0 + c * x) ** 3
      break if v > 0
    end
    u = rand
    if u < 1.0 - 0.0331 * (x*x) * (x*x) ||
       Math.log(u) < 0.5*x*x + d*(1.0 - v + Math.log(v))
      return d * v
    end
  end
end

def rand_normal
  # Box-Muller
  u1 = rand; u2 = rand
  Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
end

def rdirichlet(alphas)
  gs = alphas.map { |a| rgamma(a) }
  s = gs.sum
  gs.map { |g| g / s }
end

topics = [
  [0,1,2], [3,4,5], [6,7,8],
  [9,10,11],[12,13,14],[15,16,17]
]
alpha_sparse = [0.1, 0.1, 0.1]
alpha_dense  = [0.3, 0.3, 0.3]

docs = []
50.times do
  theta = rdirichlet(alpha_sparse)
  len = 100
  ws = len.times.map {
    u = rand; cum = 0.0; t = 0
    theta.each_with_index { |p,i| cum += p; (t = i; break) if u <= cum }
    topics[t][rand(3)]
  }
  docs << [1850, len, ws]
end
500.times do
  theta = rdirichlet(alpha_dense)
  len = 200
  ws = len.times.map {
    u = rand; cum = 0.0; t = 0
    theta.each_with_index { |p,i| cum += p; (t = i; break) if u <= cum }
    topics[3+t][rand(3)]
  }
  docs << [1950, len, ws]
end

FileUtils.mkdir_p("test_data")
File.write("test_data/vocab.txt", words.join("\n") + "\n")
File.write("test_data/metadata.txt", "num_documents=#{docs.size}\nvocab_size=#{V}\nyear_min=1850\nyear_max=1950\n")
File.open("test_data/documents.txt","w") do |f|
  docs.each { |y,l,ws| f.puts "#{y} #{l} #{ws.join(" ")}" }
end
puts "Generated #{docs.size} docs"
