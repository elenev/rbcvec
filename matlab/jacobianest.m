function J = jacobianest( f, x)

n = length(x);
J = zeros(n);

h = 1e-7;
transpose=0;

for i=1:n
    dx = zeros(size(x));
    dx(i) = h;
    fxplus = f(x+dx);
    fxminus = f(x-dx);
    if size(fxplus,2)>1
        fxplus=fxplus';
        fxminus=fxminus';
        transpose=1;
    end
    if size(dx,2)>1
       dx=dx'; 
    end
    df = fxplus - fxminus;
    J(:,i) = df / (2 * dx(i));
end

J = J(1:length(df),:);

if transpose==1
   J=J'; 
end

%while det(J) < 1e-10
%     h = 10*h;
% 
%     for i=1:n
%         dx = zeros(n,1);
%         dx(i) = h;
%         df = f(x+dx) - f(x-dx);
%         J(:,i) = df / (2 * dx(i));
%     end
% end

end

